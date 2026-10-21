#include "MSDFManager.h"
#include "MSDFCache.h"

// Placeholder arena (VirtualAlloc2 / MapViewOfFile3) removed.
// Blocks are now mapped with plain MapViewOfFile sized to the real file,
// and evicted one at a time (LRU) instead of flushing every slot.

MSDFManager::MSDFManager() {
    GetSystemInfo(&s_si);
}

MSDFManager::~MSDFManager() {
    FlushAll();
}

void MSDFManager::FreeBlock(uint32_t blockIndex) {
    if (blockIndex >= MAX_MAPPED_BLOCKS) return;

    MappedBlock& block = s_mappedBlocks[blockIndex];
    MSDFCache::BlockKey keyToErase = block.key;

    block.Reset();
    s_blockCache.erase(keyToErase);
    s_freeMask |= (1ULL << blockIndex);

    if (s_lastBlockIndex == blockIndex) {
        s_lastBlockIndex = 0xFFFFFFFF;
        s_lastBlockKey = {};
    }
}

void MSDFManager::FreeBlockByKey(MSDFCache::BlockKey key) {
    auto it = s_blockCache.find(key);
    if (it == s_blockCache.end()) return;
    FreeBlock(it->second);
}

void MSDFManager::FlushAll() {
    for (auto& block : s_mappedBlocks) {
        block.Reset();
    }
    s_blockCache.clear();
    s_freeMask = (1ULL << MAX_MAPPED_BLOCKS) - 1;

    s_lastBlockIndex = 0xFFFFFFFF;
    s_lastBlockKey = {};
}

uint32_t MSDFManager::FindLRUSlot() {
    uint32_t lruIndex = 0xFFFFFFFF;
    uint64_t lruStamp = UINT64_MAX;
    for (uint32_t i = 0; i < MAX_MAPPED_BLOCKS; ++i) {
        const MappedBlock& b = s_mappedBlocks[i];
        if (b.IsLoaded() && b.lruStamp < lruStamp) {
            lruStamp = b.lruStamp;
            lruIndex = i;
        }
    }
    return lruIndex;
}

uint32_t MSDFManager::RegisterFont(FontHash hash) {
    auto it = s_fontHashToId.find(hash);
    if (it != s_fontHashToId.end()) {
        return it->second;
    }

    uint32_t fontId = s_nextFontId++;
    s_fontHashToId.emplace(hash, fontId);
    s_fontIdToHash.emplace(fontId, hash);
    return fontId;
}

FontHash MSDFManager::GetFontHash(uint32_t fontId) {
    auto it = s_fontIdToHash.find(fontId);
    return (it != s_fontIdToHash.end()) ? it->second : 0;
}

bool MSDFManager::LoadMappedBlock(const MSDFCache::BlockWrap& wrap, MappedBlock& outBlock) {
    outBlock.file.handle = CreateFileW(wrap.path.native().c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (outBlock.file.handle == INVALID_HANDLE_VALUE) {
        outBlock.Reset();
        return false;
    }

    LARGE_INTEGER fileSizeLI;
    if (!GetFileSizeEx(outBlock.file.handle, &fileSizeLI) || fileSizeLI.QuadPart <= 0) {
        outBlock.Reset();
        return false;
    }
    outBlock.fileSize = static_cast<uint64_t>(fileSizeLI.QuadPart);

    // Sanity cap: a block file can never legitimately exceed this.
    if (outBlock.fileSize > MSDFCache::MAX_SAFE_ALLOCATION) {
        outBlock.Reset();
        return false;
    }

    // Size 0/0 = map the whole file; address space used == real file size.
    outBlock.mapping.handle = CreateFileMappingW(outBlock.file.handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!outBlock.mapping.handle) {
        outBlock.Reset();
        return false;
    }

    outBlock.view.ptr = MapViewOfFile(outBlock.mapping.handle, FILE_MAP_READ, 0, 0, 0);
    if (!outBlock.view.ptr) {
        outBlock.Reset();
        return false;
    }

    if (outBlock.fileSize < sizeof(MSDFCache::BlockFileHeader)) {
        outBlock.Reset();
        return false;
    }

    outBlock.header = static_cast<const MSDFCache::BlockFileHeader*>(outBlock.view.ptr);
    if (outBlock.header->magic != MSDFCache::BLOCK_MAGIC ||
        outBlock.header->version != MSDFCache::CACHE_VERSION ||
        outBlock.header->blockId != wrap.key.blockId ||
        outBlock.header->entryCount > MSDFCache::BLOCK_SIZE) {
        outBlock.Reset();
        return false;
    }

    outBlock.entryCount = outBlock.header->entryCount;
    outBlock.entries = reinterpret_cast<const MSDFCache::GlyphEntry*>(
        static_cast<const uint8_t*>(outBlock.view.ptr) + sizeof(MSDFCache::BlockFileHeader));

    size_t hashTableOffset = sizeof(MSDFCache::BlockFileHeader) +
        static_cast<size_t>(outBlock.entryCount) * sizeof(MSDFCache::GlyphEntry);
    outBlock.hashTable = reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>(outBlock.view.ptr) + hashTableOffset);

    size_t payloadOffset = hashTableOffset + (MSDFCache::BLOCK_SIZE * sizeof(uint32_t));
    if (outBlock.fileSize < payloadOffset) {
        outBlock.Reset();
        return false;
    }
    outBlock.payload = static_cast<const uint8_t*>(outBlock.view.ptr) + payloadOffset;

    size_t maxPayload = static_cast<size_t>(outBlock.fileSize - payloadOffset);
    for (uint32_t i = 0; i < outBlock.entryCount; ++i) {
        const MSDFCache::GlyphEntry& e = outBlock.entries[i];
        if (e.dataSize > 0 && static_cast<uint64_t>(e.dataOffset) + e.dataSize > maxPayload) {
            outBlock.Reset();
            return false;
        }
    }
    outBlock.key = wrap.key;

    return true;
}

MSDFManager::MappedBlock* MSDFManager::GetOrLoadMappedBlock(const MSDFCache::BlockWrap& wrap) {
    // Fast path: same block as last call.
    if (s_lastBlockIndex != 0xFFFFFFFF && s_lastBlockKey == wrap.key) {
        MappedBlock& b = s_mappedBlocks[s_lastBlockIndex];
        b.lruStamp = ++s_lruClock;
        return &b;
    }

    auto it = s_blockCache.find(wrap.key);
    if (it != s_blockCache.end()) {
        s_lastBlockIndex = it->second;
        s_lastBlockKey = wrap.key;
        MappedBlock& b = s_mappedBlocks[it->second];
        b.lruStamp = ++s_lruClock;
        return &b;
    }

    // No free slot: evict only the least-recently-used block (no full flush).
    if (s_freeMask == 0) {
        uint32_t victim = FindLRUSlot();
        if (victim == 0xFFFFFFFF) return nullptr;
        FreeBlock(victim);
    }

    uint32_t slotIndex = static_cast<uint32_t>(std::countr_zero(s_freeMask));
    if (slotIndex >= MAX_MAPPED_BLOCKS) return nullptr;

    MappedBlock& newBlock = s_mappedBlocks[slotIndex];
    if (!LoadMappedBlock(wrap, newBlock)) return nullptr;

    s_freeMask &= ~(1ULL << slotIndex);
    newBlock.lruStamp = ++s_lruClock;

    s_lastBlockIndex = slotIndex;
    s_lastBlockKey = wrap.key;

    s_blockCache[wrap.key] = slotIndex;
    return &newBlock;
}

bool MSDFManager::LoadGlyph(const MSDFCache::BlockWrap& wrap, uint32_t codepoint, GlyphMetrics& outMetrics) {
    MappedBlock* blockPtr = GetOrLoadMappedBlock(wrap);
    if (!blockPtr) return false;

    uint32_t entryIndex = blockPtr->hashTable[codepoint & (MSDFCache::BLOCK_SIZE - 1)];
    if (entryIndex == 0xFFFFFFFF || entryIndex >= blockPtr->entryCount) return false;

    const MSDFCache::GlyphEntry& ge = blockPtr->entries[entryIndex];
    if (ge.codepoint != codepoint) return false;

    outMetrics.width = ge.width;
    outMetrics.height = ge.height;
    outMetrics.bitmapTop = ge.bitmapTop;
    outMetrics.bitmapLeft = ge.bitmapLeft;
    outMetrics.pixelData = ge.dataSize > 0 ? blockPtr->payload + ge.dataOffset : nullptr;
    outMetrics.msdf = (ge.flags & MSDFCache::GLYPH_FLAG_SDF_ONLY) == 0;

    return true;
}
