#pragma once
#include "MSDF.h"
#include "MSDFUtils.h"
#include "ankerl/unordered_dense.h"
#include <filesystem>
#include <deque>
#include <atomic>

class MSDFManager;
class MSDFPregen;
class MSDFFont;

class MSDFCache {
    struct BlockKey {
        uint32_t fontId;
        uint32_t blockId;

        BlockKey() : fontId(0xFFFFFFFF), blockId(0xFFFFFFFF) {}
        BlockKey(uint32_t font, uint32_t block) : fontId(font), blockId(block) {}

        bool operator==(const BlockKey& other) const {
            return fontId == other.fontId && blockId == other.blockId;
        }
        uint64_t pack() const { return (static_cast<uint64_t>(fontId) << 32) | blockId; }
    };

    friend class MSDFFont;
    friend class MSDFPregen;
    friend class MSDFManager;
    friend struct std::hash<BlockKey>;

public:
    MSDFCache(const FT_Byte* fontData, FT_Long dataSize,
        const char* familyName, const char* styleName,
        uint32_t sdfRenderSize, uint32_t sdfSpread);
    ~MSDFCache();

    MSDFCache(const MSDFCache&) = delete;
    MSDFCache& operator=(const MSDFCache&) = delete;
    MSDFCache(MSDFCache&&) = delete;
    MSDFCache& operator=(MSDFCache&&) = delete;

private:
    static constexpr auto* CACHE_DIR = "LEXARA";
    static constexpr uint32_t CACHE_VERSION = 1;
    static constexpr uint32_t BLOCK_MAGIC = 0x4D534442;
    static constexpr uint32_t MANIFEST_MAGIC = 0x4D534D46;
    static constexpr size_t WRITE_BATCH_SIZE = 512; // larger = fewer synchronous disk writes on the render thread (flushed on shutdown)
    static constexpr size_t BLOCK_SIZE = 512;
    static constexpr size_t MAX_SAFE_ALLOCATION = 32 * 1024 * 1024;

    struct CacheKey {
        uint32_t sdfRenderSize = 0;
        uint32_t sdfSpread = 0;
        bool operator==(const CacheKey& other) const {
            return sdfRenderSize == other.sdfRenderSize &&
                sdfSpread == other.sdfSpread;
        }
    };

    struct BlockWrap {
        BlockKey key;
        std::filesystem::path path;
    };

#pragma pack(push, 1)
    struct ManifestHeader {
        uint32_t magic;
        uint32_t version;
        CacheKey key;
        uint32_t entryCount;
        uint32_t pad;
    };

    struct ManifestEntry {
        uint32_t codepoint;
        uint32_t blockId;
    };

    struct alignas(64) BlockFileHeader {
        uint32_t magic;
        uint32_t version;
        uint32_t blockId;
        uint32_t entryCount;
    };

    struct alignas(64) GlyphEntry {
        uint32_t codepoint;
        uint16_t width;
        uint16_t height;
        FT_Int bitmapTop;
        FT_Int bitmapLeft;
        uint32_t dataOffset;
        uint32_t dataSize;

        bool operator<(const GlyphEntry& other) const {
            return codepoint < other.codepoint;
        }
    };
#pragma pack(pop)

    static_assert(sizeof(ManifestHeader) == 24);
    static_assert(sizeof(ManifestEntry) == 8);
    static_assert(sizeof(BlockFileHeader) == 64);
    static_assert(sizeof(GlyphEntry) == 64);

    bool TryLoadGlyph(uint32_t codepoint, GlyphMetrics& outMetrics);
    bool StoreGlyph(GlyphMetricsToStore&& metrics);
    size_t GetManifestSize();
    size_t CollectCachedCodepoints(std::vector<uint32_t>& out, size_t maxCount, size_t& inoutIndex);

    using ManifestMap = ankerl::unordered_dense::map<uint32_t, ManifestEntry>;

    bool LoadManifest();
    bool SaveManifest(bool isLocked = false);
    bool LoadManifestFromFile(const std::filesystem::path& path, ManifestMap& outMap) const;
    static bool LoadManifestJournal(const std::filesystem::path& journalPath, ManifestMap& outMap, size_t& outEntriesApplied);
    bool AppendManifestJournal(const std::vector<ManifestEntry>& entries);

    void BuildBlockLockPath(uint32_t blockId, std::filesystem::path& outPath) const;
    void BuildBlockPath(uint32_t blockId, std::filesystem::path& outPath) const;

    bool FlushPendingWrites();

    // ---- Background (asynchronous) writes -------------------------------------------------
    // Runtime caches write to disk on a worker thread so the render thread never waits on file I/O.
    // The worker only touches files and its own buffers; all shared state (manifest, mapped blocks)
    // stays on the render thread, which applies finished jobs in DrainAsyncResults().
    // Pre-generation keeps the original synchronous path (m_async == false).
    void SetAsyncWrites(bool enable);

    struct AsyncJob {
        MSDFCache* owner = nullptr;
        uint32_t blockId = 0;
        std::filesystem::path blockPath;
        std::filesystem::path blockLockPath;
        std::filesystem::path manifestLockPath;
        std::filesystem::path journalPath;
        std::vector<GlyphMetricsToStore> glyphs;
    };
    struct AsyncResult {
        MSDFCache* owner = nullptr;
        uint32_t blockId = 0;
        std::vector<ManifestEntry> entries; // empty if the job failed
    };
    struct AsyncState; // defined in MSDFCache.cpp

    static AsyncState& GetAsyncState();
    static void WorkerMain();
    static bool ExecuteJob(AsyncJob& job, std::vector<ManifestEntry>& outEntries);
    static bool ReadWholeFile(const std::filesystem::path& path, std::vector<uint8_t>& out);
    static bool AppendJournalFile(const std::filesystem::path& path, const std::vector<ManifestEntry>& entries);
    static void DrainAsyncResults();

    // inlineMode: run the jobs on the calling thread (used at process exit, when the worker may already be gone)
    bool FlushPendingWritesAsync(bool inlineMode = false);
    void WaitForAsyncIdle();

    bool m_async = false;
    std::atomic<uint32_t> m_inflightJobs{ 0 };
    // blocks with a write in flight: their glyphs are not read from disk until the write lands
    ankerl::unordered_dense::map<uint32_t, uint32_t> m_inflightBlocks;
    bool WriteBlockFile(uint32_t blockId, std::vector<GlyphMetricsToStore*>& pending, std::vector<ManifestEntry>& outEntries);
    void CleanupOrphans() const;

    static uint32_t GetBlockId(uint32_t codepoint);
    static std::filesystem::path GetCacheBasePath(const char* familyName, const char* styleName,
        uint32_t sdfRenderSize, uint32_t sdfSpread);
    static std::string SanitizeName(std::string_view name);

    std::filesystem::path m_cacheBasePath;
    std::filesystem::path m_cacheManifestPath;
    std::filesystem::path m_cacheManifestLockPath;
    std::filesystem::path m_cacheManifestJournalPath;

    CacheKey m_key;
    ManifestMap m_manifest;

    bool m_manifestLoaded = false;
    uint32_t m_fontID = 0xFFFFFFFF;

    VectorPool<uint8_t> m_vecPool;
    VectorPool<uint32_t> m_hashPool;
    VectorPool<GlyphEntry> m_gEntryPool;
    VectorPool<ManifestEntry> m_mEntryPool;

    std::deque<GlyphMetricsToStore> m_pendingWrites;
    
    ankerl::unordered_dense::map<uint32_t, BlockWrap> m_blockWrap;

    static MSDFManager s_manager;
};

template<>
struct std::hash<MSDFCache::BlockKey> {
    size_t operator()(const MSDFCache::BlockKey& k) const noexcept {
        uint64_t packed = k.pack();
        return ankerl::unordered_dense::detail::wyhash::hash(&packed, sizeof(packed));
    }
};
