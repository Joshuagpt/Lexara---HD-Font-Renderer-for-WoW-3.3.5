#include "MSDFCache.h"
#include "MSDFManager.h"
#include "../ShutdownCheck.h"
#include <fstream>
#include <ranges>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>

MSDFManager MSDFCache::s_manager = MSDFManager();

MSDFCache::MSDFCache(const FT_Byte* fontData, FT_Long dataSize, const char* familyName, const char* styleName,
    uint32_t sdfRenderSize, uint32_t sdfSpread)
    : m_key{ .sdfRenderSize = sdfRenderSize, .sdfSpread = sdfSpread }
{
    m_cacheBasePath = GetCacheBasePath(familyName, styleName, sdfRenderSize, sdfSpread);
    m_cacheManifestPath = m_cacheBasePath / "manifest.dat";
    m_cacheManifestLockPath = m_cacheBasePath / "manifest.lock";
    m_cacheManifestJournalPath = m_cacheBasePath / "manifest.jrn";

    m_fontID = MSDFManager::RegisterFont(HashFont(fontData, dataSize));

    std::error_code ec;
    std::filesystem::create_directories(m_cacheBasePath, ec);
}

MSDFCache::~MSDFCache() {
    if (m_async && g_isProcessTerminating) {
        // Process exit: the worker thread may already have been killed, so never wait for it.
        // Write whatever is still pending right here on the calling thread.
        FlushPendingWritesAsync(true);
        SetAsyncWrites(false);
    }
    else {
        FlushPendingWrites();
        if (m_async) {
            WaitForAsyncIdle();
            SetAsyncWrites(false); // unregisters this cache from the worker's result routing

            // Writes only append to the journal while running; compact it into the manifest once it grows.
            std::error_code ec;
            if (m_manifestLoaded) {
                auto jsize = std::filesystem::file_size(m_cacheManifestJournalPath, ec);
                if (!ec && jsize > 128 * 1024) SaveManifest(false);
            }
        }
    }
    CleanupOrphans();
    MSDFManager::FlushAll();
    m_vecPool.TrimAll();
    m_mEntryPool.TrimAll();
}

std::string MSDFCache::SanitizeName(std::string_view name) {
    if (name.empty()) return "unnamed";
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        if (std::string_view("/:*?\"<>|\\").find(c) != std::string_view::npos || std::iscntrl(static_cast<unsigned char>(c))) {
            out.push_back('_');
        }
        else {
            out.push_back(c);
        }
    }
    while (!out.empty() && (out.back() == ' ' || out.back() == '.')) out.pop_back();
    return out.empty() ? "unnamed" : out;
}

std::filesystem::path MSDFCache::GetCacheBasePath(const char* familyName, const char* styleName,
    uint32_t sdfRenderSize, uint32_t sdfSpread) {
    std::string fam = SanitizeName(familyName);
    std::string sty = SanitizeName(styleName);
    std::string folderName = fam + "_" + sty + "_s" + std::to_string(sdfRenderSize) + "_sp" + std::to_string(sdfSpread);

    // Preferred location: <folder containing wow.exe>\LEXARA\<font folder>.
    // Falls back to %TEMP% if that folder cannot be created or written (e.g. read-only install location).
    wchar_t exePath[MAX_PATH * 2] = {};
    const DWORD n = GetModuleFileNameW(nullptr, exePath, static_cast<DWORD>(std::size(exePath)));
    if (n > 0 && n < std::size(exePath)) {
        std::filesystem::path base = std::filesystem::path(exePath).parent_path() / CACHE_DIR / folderName;
        std::error_code ec;
        std::filesystem::create_directories(base, ec);
        if (!ec) {
            // Make sure we can actually write there.
            HANDLE h = CreateFileW((base / L".write_test").c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
            if (h != INVALID_HANDLE_VALUE) {
                CloseHandle(h);
                return base;
            }
        }
    }

    wchar_t tempPath[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tempPath);
    return std::filesystem::path(tempPath) / CACHE_DIR / folderName;
}

void MSDFCache::BuildBlockLockPath(uint32_t blockId, std::filesystem::path& outPath) const {
    char buf[32];
    snprintf(buf, sizeof(buf), "block_%u.lock", blockId);
    outPath = m_cacheBasePath / buf;
}

void MSDFCache::BuildBlockPath(uint32_t blockId, std::filesystem::path& outPath) const {
    char buf[32];
    snprintf(buf, sizeof(buf), "block_%u.dat", blockId);
    outPath = m_cacheBasePath / buf;
}

uint32_t MSDFCache::GetBlockId(uint32_t codepoint) {
    return codepoint >> static_cast<uint32_t>(std::countr_zero(BLOCK_SIZE));
}

bool MSDFCache::TryLoadGlyph(uint32_t codepoint, GlyphMetrics& outMetrics) {
    if (m_async) DrainAsyncResults();
    auto mit = m_manifest.find(codepoint);
    if (mit == m_manifest.end()) return false;
    // The block file is being replaced by the worker: report a miss (the glyph is simply regenerated).
    if (m_async && !m_inflightBlocks.empty() && m_inflightBlocks.contains(mit->second.blockId)) return false;
    auto bit = m_blockWrap.find(mit->second.blockId);
    if (bit != m_blockWrap.end()) {
        return MSDFManager::LoadGlyph(bit->second, codepoint, outMetrics);
    }
    uint32_t blockId = mit->second.blockId;
    BlockKey block(m_fontID, blockId);
    std::filesystem::path blockPath;
    BuildBlockPath(blockId, blockPath);
    BlockWrap wrap = { .key = block, .path = blockPath };
    m_blockWrap[mit->second.blockId] = wrap;
    return MSDFManager::LoadGlyph(wrap, codepoint, outMetrics);
}

bool MSDFCache::StoreGlyph(GlyphMetricsToStore&& metrics) {
    if (m_async) DrainAsyncResults();
    if (!m_manifestLoaded) {
        if (!LoadManifest()) return false;
    }
    m_pendingWrites.push_back(std::move(metrics));
    if (m_pendingWrites.size() >= WRITE_BATCH_SIZE) {
        FlushPendingWrites();
    }
    return true;
}

bool MSDFCache::LoadManifest() {
    ScopedFileLock lock;
    if (!lock.AcquireShared(m_cacheManifestLockPath, 10000)) {
        return false;
    }

    std::error_code ec;
    bool pathExists = std::filesystem::exists(m_cacheManifestPath, ec);
    bool journalExists = std::filesystem::exists(m_cacheManifestJournalPath, ec);

    if (!pathExists && !journalExists) {
        m_manifestLoaded = true;
        return true;
    }

    if (pathExists) {
        auto fsize = std::filesystem::file_size(m_cacheManifestPath, ec);
        if (!ec && fsize > sizeof(ManifestHeader) && fsize < MAX_SAFE_ALLOCATION) {
            size_t estimatedEntries = (fsize - sizeof(ManifestHeader)) / sizeof(ManifestEntry);
            m_manifest.reserve(std::min(estimatedEntries + estimatedEntries / 10, static_cast<uint32_t>(0x110000))); // 1,114,112 - max unicode range
        }
        if (!LoadManifestFromFile(m_cacheManifestPath, m_manifest)) {
            m_manifest.clear();
            return false;
        }
    }

    size_t applied = 0;
    LoadManifestJournal(m_cacheManifestJournalPath, m_manifest, applied);

    m_manifestLoaded = true;
    return true;
}

bool MSDFCache::LoadManifestFromFile(const std::filesystem::path& path, ManifestMap& outMap) const {
    std::error_code ec;
    auto fsize = std::filesystem::file_size(path, ec);
    if (ec || fsize < sizeof(ManifestHeader) || fsize > MAX_SAFE_ALLOCATION) return false;

    FileGuard file(CreateFileW(path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
    if (!file.IsValid()) return false;

    FileGuard mapping(CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr));
    if (!mapping.IsValid()) return false;

    ViewGuard view(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
    if (!view) return false;

    const auto* hdr = static_cast<const ManifestHeader*>(view.ptr);
    if (hdr->magic == MANIFEST_MAGIC && hdr->version == CACHE_VERSION && hdr->key == m_key) {
        uint64_t expectedSize = sizeof(ManifestHeader) + static_cast<uint64_t>(hdr->entryCount) * sizeof(ManifestEntry);
        if (fsize >= expectedSize) {
            auto* entries = reinterpret_cast<const ManifestEntry*>(
                static_cast<const uint8_t*>(view.ptr) + sizeof(ManifestHeader));
            for (uint32_t i = 0; i < hdr->entryCount; ++i) {
                outMap.try_emplace(entries[i].codepoint, entries[i]);
            }
            return true;
        }
    }
    return false;
}

bool MSDFCache::LoadManifestJournal(const std::filesystem::path& journalPath, ManifestMap& outMap, size_t& outEntriesApplied) {
    outEntriesApplied = 0;
    std::error_code ec;
    if (!std::filesystem::exists(journalPath, ec) || ec) return true;

    auto jsize = std::filesystem::file_size(journalPath, ec);
    if (ec || jsize == 0) return !ec;

    if (jsize % sizeof(ManifestEntry) != 0) return false;
    if (jsize > MAX_SAFE_ALLOCATION) return false;

    std::ifstream in(journalPath, std::ios::binary);
    if (!in.good()) return false;

    ManifestEntry e;
    while (in.read(reinterpret_cast<char*>(&e), sizeof(ManifestEntry))) {
        outMap[e.codepoint] = e;
        ++outEntriesApplied;
    }
    return true;
}

bool MSDFCache::AppendManifestJournal(const std::vector<ManifestEntry>& entries) {
    if (entries.empty()) return true;

    std::error_code ec;
    std::filesystem::create_directories(m_cacheBasePath, ec);
    if (ec) return false;

    FileGuard file(CreateFileW(m_cacheManifestJournalPath.c_str(), FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!file.IsValid()) return false;
    if (entries.size() > UINT32_MAX / sizeof(ManifestEntry)) return false;

    DWORD totalBytes = static_cast<DWORD>(entries.size() * sizeof(ManifestEntry));
    auto buffer = m_vecPool.AcquireSized(totalBytes);
    std::memcpy(buffer.data(), entries.data(), totalBytes);

    DWORD written = 0;
    bool ok = WriteFile(file, buffer.data(), totalBytes, &written, nullptr) && (written == totalBytes);
    if (ok) {
        FlushFileBuffers(file);
        file.successful = true;
    }

    m_vecPool.Release(std::move(buffer));
    return ok;
}

bool MSDFCache::SaveManifest(bool isLocked) {
    ScopedFileLock lock;
    if (!isLocked && !lock.AcquireExclusive(m_cacheManifestLockPath, 1000)) return false;

    std::filesystem::path tmpManifest = m_cacheManifestPath;
    tmpManifest.replace_extension(".tmp");

    FileGuard file(CreateFileW(tmpManifest.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, nullptr));
    if (file.handle == INVALID_HANDLE_VALUE) return false;

    ManifestHeader hdr{ .magic = MANIFEST_MAGIC, .version = CACHE_VERSION, .key = m_key, .entryCount = m_manifest.size(), .pad = 0 };
    DWORD written = 0;

    if (!WriteFile(file.handle, &hdr, sizeof(hdr), &written, nullptr)) return false;

    auto entries = m_mEntryPool.Acquire(m_manifest.size());
    entries.reserve(m_manifest.size());

    FinalAction cleanup([&]() {
        if (!entries.empty()) m_mEntryPool.Release(std::move(entries));
        });

    for (const auto& kv : m_manifest) {
        entries.push_back({ .codepoint =  kv.first, .blockId = kv.second.blockId });
    }

    if (!entries.empty()) {
        if (!WriteFile(file.handle, entries.data(), static_cast<DWORD>(entries.size() * sizeof(ManifestEntry)), &written, nullptr)) {
            return false;
        }
    }
    FlushFileBuffers(file.handle);
    file.Close();

    if (MoveFileExW(tmpManifest.c_str(), m_cacheManifestPath.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        std::error_code ec;
        std::filesystem::remove(m_cacheManifestJournalPath, ec);
        return true;
    }
    return false;
}

size_t MSDFCache::GetManifestSize() {
    if (!m_manifestLoaded) {
        LoadManifest();
    }
    return m_manifest.size();
}

size_t MSDFCache::CollectCachedCodepoints(std::vector<uint32_t>& out, size_t maxCount, size_t& inoutIndex) {
    if (maxCount == 0) return 0;
    if (!m_manifestLoaded) LoadManifest();
    if (!m_manifestLoaded || m_manifest.empty()) return 0;

    size_t seen = 0;
    size_t added = 0;
    for (const auto& kv : m_manifest) {
        if (seen < inoutIndex) { ++seen; continue; }
        out.push_back(kv.first);
        ++seen;
        ++added;
        if (added >= maxCount) break;
    }
    inoutIndex = seen;
    return added;
}


bool MSDFCache::FlushPendingWrites() {
    if (m_pendingWrites.empty()) return true;
    if (m_async) return FlushPendingWritesAsync();

    std::error_code ec;
    std::filesystem::create_directories(m_cacheBasePath, ec);
    if (ec) return false;

    ankerl::unordered_dense::map<uint32_t, std::vector<GlyphMetricsToStore*>> byBlock;
    for (auto& pw : m_pendingWrites) {
        byBlock[GetBlockId(pw.codepoint)].push_back(&pw);
    }
    auto newEntries = m_mEntryPool.Acquire(m_pendingWrites.size());

    size_t maxBlockSize = byBlock.empty() ? 0 : std::ranges::max(
        byBlock | std::views::values | std::views::transform([](auto& v) { return v.size(); })
    );
    auto blockEntries = m_mEntryPool.Acquire(maxBlockSize);

    for (auto& kv : byBlock) {
        uint32_t blockId = kv.first;
        std::filesystem::path lockPath;
        BuildBlockLockPath(blockId, lockPath);

        ScopedFileLock lock;
        if (!lock.AcquireExclusive(lockPath, 1000)) continue;

        blockEntries.clear();
        if (!WriteBlockFile(blockId, kv.second, blockEntries)) continue;

        newEntries.insert(newEntries.end(), blockEntries.begin(), blockEntries.end());
    }
    m_pendingWrites.clear();

    if (!newEntries.empty()) {
        ScopedFileLock lock;
        if (lock.AcquireExclusive(m_cacheManifestLockPath, 1000)) {
            if (AppendManifestJournal(newEntries)) {
                for (const ManifestEntry& me : newEntries) {
                    m_manifest[me.codepoint] = me;
                }
                SaveManifest(true);
            }
        }
    }

    m_mEntryPool.Release(std::move(blockEntries));
    m_mEntryPool.Release(std::move(newEntries));
    return true;
}

bool MSDFCache::WriteBlockFile(uint32_t blockId, std::vector<GlyphMetricsToStore*>& pending, std::vector<ManifestEntry>& outEntries) {
    std::filesystem::path blockPath;
    BuildBlockPath(blockId, blockPath);

    std::ranges::sort(pending, {}, &GlyphMetricsToStore::codepoint);

    auto bit = m_blockWrap.find(blockId);
    BlockWrap wrap;
    if (bit != m_blockWrap.end()) {
        wrap = bit->second;
    }
    else {
        BlockKey block(m_fontID, blockId);
        wrap = { .key = block, .path = blockPath };
        m_blockWrap[blockId] = wrap;
    }
    MSDFManager::MappedBlock* cachedBlock = MSDFManager::GetOrLoadMappedBlock(wrap);

    uint32_t oldEntriesCount = cachedBlock ? cachedBlock->entryCount : 0;
    auto mergedEntries = m_gEntryPool.Acquire(oldEntriesCount + pending.size());
    auto hashTable = m_hashPool.AcquireSized(BLOCK_SIZE, 0xFFFFFFFF);

    std::vector<uint8_t>* payloadRef = nullptr;
    FinalAction cleanup([&]() {
        if (!hashTable.empty()) m_hashPool.Release(std::move(hashTable));
        if (!mergedEntries.empty()) m_gEntryPool.Release(std::move(mergedEntries));
        if (payloadRef && !payloadRef->empty()) m_vecPool.Release(std::move(*payloadRef));
        });

    uint32_t oldIdx = 0;
    auto pendingIt = pending.begin();
    while (oldIdx < oldEntriesCount || pendingIt != pending.end()) {
        if (oldIdx < oldEntriesCount && (pendingIt == pending.end() || ((cachedBlock->entries[oldIdx].codepoint) < ((*pendingIt)->codepoint)))) {
            mergedEntries.push_back(cachedBlock->entries[oldIdx++]);
        }   
        else if (pendingIt != pending.end() && (oldIdx == oldEntriesCount || (*pendingIt)->codepoint < cachedBlock->entries[oldIdx].codepoint)) {
            auto* p = *pendingIt++;
            mergedEntries.push_back({
                .codepoint = p->codepoint,
            	.width = p->width,
            	.height = p->height,
            	.bitmapTop = p->bitmapTop,
            	.bitmapLeft = p->bitmapLeft,
            	.dataOffset = 0,
            	.dataSize = p->dataSize });
        }
        else {
            auto* p = *pendingIt++;
            mergedEntries.push_back({
                .codepoint = p->codepoint,
                .width = p->width,
                .height = p->height,
                .bitmapTop = p->bitmapTop,
                .bitmapLeft = p->bitmapLeft,
                .dataOffset = 0,
                .dataSize = p->dataSize });
            oldIdx++;
        }
    }
    if (mergedEntries.size() > BLOCK_SIZE) return false;

    uint32_t totalPayloadSize = 0;
    for (auto& ge : mergedEntries) {
        ge.dataOffset = totalPayloadSize;
        totalPayloadSize += ge.dataSize;
    }

    for (size_t i = 0; i < mergedEntries.size(); ++i) {
        hashTable[mergedEntries[i].codepoint & (BLOCK_SIZE - 1)] = static_cast<uint32_t>(i);
    }

    auto payloadBuffer = m_vecPool.AcquireSized(totalPayloadSize);
    payloadRef = &payloadBuffer;

    auto pendingCopyIt = pending.begin();
    for (auto& ge : mergedEntries) {
        const uint8_t* src = nullptr;
        while (pendingCopyIt != pending.end() && (*pendingCopyIt)->codepoint < ge.codepoint) {
            ++pendingCopyIt;
        }
        if (pendingCopyIt != pending.end() && (*pendingCopyIt)->codepoint == ge.codepoint) {
            src = (*pendingCopyIt)->ownedPixelData.data();
        }
        else if (cachedBlock) {
            uint32_t oldEntryIndex = cachedBlock->hashTable[ge.codepoint & (BLOCK_SIZE - 1)];
            if (oldEntryIndex != 0xFFFFFFFF && oldEntryIndex < cachedBlock->entryCount) {
                if (cachedBlock->entries[oldEntryIndex].codepoint == ge.codepoint) {
                    src = cachedBlock->payload + cachedBlock->entries[oldEntryIndex].dataOffset;
                }
            }
        }
        if (src && ge.dataSize > 0) {
            std::memcpy(payloadBuffer.data() + ge.dataOffset, src, ge.dataSize);
        }
    }
    MSDFManager::FreeBlockByKey(wrap.key);

    std::filesystem::path tmpPath = blockPath;
    tmpPath.replace_extension(".tmp");

    {
        FileGuard tmpFile(CreateFileW(tmpPath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
        if (tmpFile.handle == INVALID_HANDLE_VALUE) return false;

        BlockFileHeader bHdr{ .magic = BLOCK_MAGIC, .version = CACHE_VERSION, .blockId = blockId, .entryCount = mergedEntries.size() };
        size_t dataSize = sizeof(bHdr) + (mergedEntries.size() * sizeof(GlyphEntry)) +
            (BLOCK_SIZE * sizeof(uint32_t)) + payloadBuffer.size();

        const size_t align = MSDFManager::s_si.dwAllocationGranularity;
        size_t paddedSize = ((dataSize + align - 1) / align) * align;
        size_t paddingNeeded = paddedSize - dataSize;

        if (paddingNeeded > 0) {
            payloadBuffer.resize(payloadBuffer.size() + paddingNeeded, 0);
        }

        DWORD written;
        if (!WriteFile(tmpFile.handle, &bHdr, sizeof(bHdr), &written, nullptr) || written != sizeof(bHdr)) {
            return false;
        }
        if (!WriteFile(tmpFile.handle, mergedEntries.data(),
            static_cast<DWORD>(mergedEntries.size() * sizeof(GlyphEntry)), &written, nullptr)) {
            return false;
        }
        if (!WriteFile(tmpFile.handle, hashTable.data(),
            static_cast<DWORD>(BLOCK_SIZE * sizeof(uint32_t)), &written, nullptr)) {
            return false;
        }
        if (!WriteFile(tmpFile.handle, payloadBuffer.data(),
            static_cast<DWORD>(payloadBuffer.size()), &written, nullptr)) {
            return false;
        }
        FlushFileBuffers(tmpFile.handle);
    }

    if (!MoveFileExW(tmpPath.c_str(), blockPath.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        std::filesystem::path oldPath = blockPath;
        oldPath.replace_extension(".old");

        if (MoveFileExW(tmpPath.c_str(), oldPath.c_str(), MOVEFILE_REPLACE_EXISTING) || std::filesystem::exists(oldPath)) {
            auto mwit = m_blockWrap.find(blockId);
            if (mwit != m_blockWrap.end()) mwit->second.path = oldPath;
        }
        else {
            return false;
        }
    }

    for (auto* pw : pending) {
        outEntries.push_back({ .codepoint = pw->codepoint, .blockId = blockId });
    }

    return true;
}

void MSDFCache::CleanupOrphans() const {
    std::error_code ec;
    if (!std::filesystem::exists(m_cacheBasePath, ec)) return;

    for (const auto& entry : std::filesystem::directory_iterator(m_cacheBasePath, ec)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".old" || ext == ".tmp") {
                std::filesystem::remove(entry.path(), ec);
            }
        }
    }
}


// ============================================================================
// Background writes
// ============================================================================

struct MSDFCache::AsyncState {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<AsyncJob> jobs;
    std::vector<AsyncResult> results;
    std::unordered_set<MSDFCache*> liveOwners; // caches that may still receive results
    std::atomic<bool> hasResults{ false };
    bool workerStarted = false;
};

MSDFCache::AsyncState& MSDFCache::GetAsyncState() {
    // Intentionally leaked: the detached worker thread may outlive static destruction.
    static AsyncState* state = new AsyncState();
    return *state;
}

void MSDFCache::SetAsyncWrites(bool enable) {
    AsyncState& st = GetAsyncState();
    std::lock_guard<std::mutex> lk(st.mtx);
    if (enable) st.liveOwners.insert(this);
    else st.liveOwners.erase(this);
    m_async = enable;
}

void MSDFCache::WorkerMain() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    AsyncState& st = GetAsyncState();
    for (;;) {
        AsyncJob job;
        {
            std::unique_lock<std::mutex> lk(st.mtx);
            st.cv.wait(lk, [&] { return !st.jobs.empty(); });
            job = std::move(st.jobs.front());
            st.jobs.pop_front();
        }

        AsyncResult res;
        res.owner = job.owner;
        res.blockId = job.blockId;
        if (!ExecuteJob(job, res.entries)) res.entries.clear();

        {
            std::lock_guard<std::mutex> lk(st.mtx);
            st.results.push_back(std::move(res));
            st.hasResults.store(true, std::memory_order_release);
        }
    }
}

bool MSDFCache::FlushPendingWritesAsync(bool inlineMode) {
    if (m_pendingWrites.empty()) return true;

    std::error_code ec;
    std::filesystem::create_directories(m_cacheBasePath, ec);
    if (ec) { m_pendingWrites.clear(); return false; }

    ankerl::unordered_dense::map<uint32_t, std::vector<GlyphMetricsToStore>> byBlock;
    for (auto& pw : m_pendingWrites) {
        byBlock[GetBlockId(pw.codepoint)].push_back(std::move(pw));
    }
    m_pendingWrites.clear();

    AsyncState& st = GetAsyncState();
    for (auto& kv : byBlock) {
        AsyncJob job;
        job.owner = this;
        job.blockId = kv.first;
        BuildBlockPath(kv.first, job.blockPath);
        BuildBlockLockPath(kv.first, job.blockLockPath);
        job.manifestLockPath = m_cacheManifestLockPath;
        job.journalPath = m_cacheManifestJournalPath;
        job.glyphs = std::move(kv.second);

        // Unmap on this (render) thread so the file can be replaced.
        // Until the job lands, TryLoadGlyph reports a miss for this block.
        MSDFManager::FreeBlockByKey(BlockKey(m_fontID, kv.first));

        if (inlineMode) {
            std::vector<ManifestEntry> ignored;
            ExecuteJob(job, ignored);
            continue;
        }

        ++m_inflightBlocks[kv.first];
        m_inflightJobs.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(st.mtx);
            st.jobs.push_back(std::move(job));
            if (!st.workerStarted) {
                st.workerStarted = true;
                std::thread(&MSDFCache::WorkerMain).detach();
            }
        }
        st.cv.notify_one();
    }
    return true;
}

void MSDFCache::DrainAsyncResults() {
    AsyncState& st = GetAsyncState();
    if (!st.hasResults.load(std::memory_order_acquire)) return;

    std::vector<AsyncResult> local;
    {
        std::lock_guard<std::mutex> lk(st.mtx);
        local.swap(st.results);
        st.hasResults.store(false, std::memory_order_relaxed);
    }
    for (AsyncResult& r : local) {
        bool alive;
        {
            std::lock_guard<std::mutex> lk(st.mtx);
            alive = st.liveOwners.contains(r.owner);
        }
        if (!alive) continue; // owner already destroyed

        MSDFCache* o = r.owner;
        for (const ManifestEntry& e : r.entries) o->m_manifest[e.codepoint] = e;

        auto it = o->m_inflightBlocks.find(r.blockId);
        if (it != o->m_inflightBlocks.end() && --it->second == 0) o->m_inflightBlocks.erase(it);
        o->m_inflightJobs.fetch_sub(1, std::memory_order_relaxed);
    }
}

void MSDFCache::WaitForAsyncIdle() {
    if (g_isProcessTerminating) return; // the worker may be gone
    const ULONGLONG start = GetTickCount64();
    while (m_inflightJobs.load(std::memory_order_relaxed) > 0 && GetTickCount64() - start < 15000) {
        DrainAsyncResults();
        if (m_inflightJobs.load(std::memory_order_relaxed) == 0) break;
        Sleep(2);
    }
    DrainAsyncResults();
}

bool MSDFCache::ReadWholeFile(const std::filesystem::path& path, std::vector<uint8_t>& out) {
    out.clear();
    FileGuard file(CreateFileW(path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
    if (!file.IsValid()) return false;

    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file.handle, &size) || size.QuadPart <= 0 ||
        static_cast<uint64_t>(size.QuadPart) > MAX_SAFE_ALLOCATION) return false;

    out.resize(static_cast<size_t>(size.QuadPart));
    size_t done = 0;
    while (done < out.size()) {
        DWORD got = 0;
        if (!ReadFile(file.handle, out.data() + done, static_cast<DWORD>(out.size() - done), &got, nullptr) || got == 0) {
            out.clear();
            return false;
        }
        done += got;
    }
    return true;
}

bool MSDFCache::AppendJournalFile(const std::filesystem::path& path, const std::vector<ManifestEntry>& entries) {
    if (entries.empty()) return true;
    FileGuard file(CreateFileW(path.c_str(), FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!file.IsValid()) return false;

    const DWORD totalBytes = static_cast<DWORD>(entries.size() * sizeof(ManifestEntry));
    DWORD written = 0;
    const bool ok = WriteFile(file.handle, entries.data(), totalBytes, &written, nullptr) && written == totalBytes;
    if (ok) FlushFileBuffers(file.handle);
    return ok;
}

// Runs on the worker thread. Touches only files and local buffers.
bool MSDFCache::ExecuteJob(AsyncJob& job, std::vector<ManifestEntry>& outEntries) {
    // Pending glyphs: sorted by codepoint, last write wins for duplicates.
    std::stable_sort(job.glyphs.begin(), job.glyphs.end(),
        [](const GlyphMetricsToStore& a, const GlyphMetricsToStore& b) { return a.codepoint < b.codepoint; });
    std::vector<const GlyphMetricsToStore*> pend;
    pend.reserve(job.glyphs.size());
    for (size_t i = 0; i < job.glyphs.size(); ++i) {
        if (i + 1 < job.glyphs.size() && job.glyphs[i + 1].codepoint == job.glyphs[i].codepoint) continue;
        pend.push_back(&job.glyphs[i]);
    }

    ScopedFileLock blockLock;
    if (!blockLock.AcquireExclusive(job.blockLockPath, 5000)) return false;

    // Read the existing block (if any) into private memory. Copy entries out with memcpy: the file
    // buffer is not 64-byte aligned, and GlyphEntry is alignas(64).
    std::vector<uint8_t> oldFile;
    std::vector<GlyphEntry> oldEntries;
    const uint8_t* oldPayload = nullptr;
    if (ReadWholeFile(job.blockPath, oldFile) && oldFile.size() >= sizeof(BlockFileHeader)) {
        BlockFileHeader hdr{};
        std::memcpy(&hdr, oldFile.data(), sizeof(hdr));
        bool ok = hdr.magic == BLOCK_MAGIC && hdr.version == CACHE_VERSION &&
            hdr.blockId == job.blockId && hdr.entryCount <= BLOCK_SIZE;
        size_t payloadOffset = 0;
        if (ok) {
            const size_t hashOffset = sizeof(BlockFileHeader) + static_cast<size_t>(hdr.entryCount) * sizeof(GlyphEntry);
            payloadOffset = hashOffset + BLOCK_SIZE * sizeof(uint32_t);
            ok = oldFile.size() >= payloadOffset;
        }
        if (ok) {
            oldEntries.resize(hdr.entryCount);
            if (hdr.entryCount) std::memcpy(oldEntries.data(), oldFile.data() + sizeof(BlockFileHeader), hdr.entryCount * sizeof(GlyphEntry));
            const size_t maxPayload = oldFile.size() - payloadOffset;
            for (const GlyphEntry& e : oldEntries) {
                if (e.dataSize > 0 && static_cast<uint64_t>(e.dataOffset) + e.dataSize > maxPayload) { ok = false; break; }
            }
        }
        if (ok) {
            oldPayload = oldFile.data() + payloadOffset;
            std::sort(oldEntries.begin(), oldEntries.end());
        }
        else {
            oldEntries.clear(); // damaged block: start over
        }
    }

    // Merge old and pending (pending wins on equal codepoints).
    struct Src { const uint8_t* ptr; };
    std::vector<GlyphEntry> merged;
    std::vector<Src> srcs;
    merged.reserve(oldEntries.size() + pend.size());
    srcs.reserve(oldEntries.size() + pend.size());
    size_t oi = 0, pi = 0;
    while (oi < oldEntries.size() || pi < pend.size()) {
        const bool takeOld = oi < oldEntries.size() &&
            (pi == pend.size() || oldEntries[oi].codepoint < pend[pi]->codepoint);
        if (takeOld) {
            merged.push_back(oldEntries[oi]);
            srcs.push_back({ oldEntries[oi].dataSize > 0 ? oldPayload + oldEntries[oi].dataOffset : nullptr });
            ++oi;
        }
        else {
            const GlyphMetricsToStore* p = pend[pi++];
            if (oi < oldEntries.size() && oldEntries[oi].codepoint == p->codepoint) ++oi; // replaced
            GlyphEntry ge{};
            ge.codepoint = p->codepoint;
            ge.width = p->width;
            ge.height = p->height;
            ge.bitmapTop = p->bitmapTop;
            ge.bitmapLeft = p->bitmapLeft;
            ge.dataOffset = 0;
            ge.dataSize = p->dataSize;
            merged.push_back(ge);
            srcs.push_back({ p->dataSize > 0 ? p->ownedPixelData.data() : nullptr });
        }
    }
    if (merged.size() > BLOCK_SIZE) return false;

    uint64_t totalPayload = 0;
    for (GlyphEntry& ge : merged) {
        ge.dataOffset = static_cast<uint32_t>(totalPayload);
        totalPayload += ge.dataSize;
    }
    if (totalPayload > MAX_SAFE_ALLOCATION) return false;

    std::vector<uint32_t> hashTable(BLOCK_SIZE, 0xFFFFFFFFu);
    for (size_t i = 0; i < merged.size(); ++i) {
        hashTable[merged[i].codepoint & (BLOCK_SIZE - 1)] = static_cast<uint32_t>(i);
    }

    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const size_t align = si.dwAllocationGranularity ? si.dwAllocationGranularity : 65536;
    const size_t dataSize = sizeof(BlockFileHeader) + merged.size() * sizeof(GlyphEntry) +
        BLOCK_SIZE * sizeof(uint32_t) + static_cast<size_t>(totalPayload);
    const size_t paddedSize = ((dataSize + align - 1) / align) * align;

    std::vector<uint8_t> payload(static_cast<size_t>(totalPayload) + (paddedSize - dataSize), 0);
    for (size_t i = 0; i < merged.size(); ++i) {
        if (srcs[i].ptr && merged[i].dataSize > 0) {
            std::memcpy(payload.data() + merged[i].dataOffset, srcs[i].ptr, merged[i].dataSize);
        }
    }

    std::filesystem::path tmpPath = job.blockPath;
    tmpPath.replace_extension(".tmp");
    {
        FileGuard tmpFile(CreateFileW(tmpPath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
        if (!tmpFile.IsValid()) return false;
        tmpFile.path = tmpPath;
        tmpFile.deleteOnFailure = true;

        BlockFileHeader bHdr{};
        bHdr.magic = BLOCK_MAGIC;
        bHdr.version = CACHE_VERSION;
        bHdr.blockId = job.blockId;
        bHdr.entryCount = static_cast<uint32_t>(merged.size());

        DWORD written = 0;
        auto put = [&](const void* data, size_t bytes) {
            return bytes == 0 || (WriteFile(tmpFile.handle, data, static_cast<DWORD>(bytes), &written, nullptr) && written == bytes);
        };
        if (!put(&bHdr, sizeof(bHdr)) ||
            !put(merged.data(), merged.size() * sizeof(GlyphEntry)) ||
            !put(hashTable.data(), hashTable.size() * sizeof(uint32_t)) ||
            !put(payload.data(), payload.size())) {
            return false;
        }
        FlushFileBuffers(tmpFile.handle);
        tmpFile.successful = true;
    }

    // The render thread unmapped this block when the job was queued and does not remap it until we finish,
    // so the replace normally succeeds at once. Another game instance may still hold a mapping: retry briefly.
    bool moved = false;
    for (int attempt = 0; attempt < 5 && !moved; ++attempt) {
        moved = MoveFileExW(tmpPath.c_str(), job.blockPath.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
        if (!moved) Sleep(40);
    }
    if (!moved) {
        DeleteFileW(tmpPath.c_str());
        return false;
    }

    outEntries.reserve(pend.size());
    for (const GlyphMetricsToStore* p : pend) {
        outEntries.push_back({ p->codepoint, job.blockId });
    }

    // Best effort: record in the journal so future sessions know about these glyphs.
    ScopedFileLock manifestLock;
    if (manifestLock.AcquireExclusive(job.manifestLockPath, 5000)) {
        AppendJournalFile(job.journalPath, outEntries);
    }
    return true;
}
