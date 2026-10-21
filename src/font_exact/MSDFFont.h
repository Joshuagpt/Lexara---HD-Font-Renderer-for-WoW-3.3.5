#pragma once
#include "MSDF.h"
#include "MSDFCache.h"
#include "../ShutdownCheck.h"
#include "MSDFAtlasTiles.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

class MSDFFont {
    friend class MSDFCache;
    friend class MSDFPregen;

private:
    struct AtlasPage {
        IDirect3DTexture9* texture = nullptr;
        ~AtlasPage() {
            if (texture && !g_isProcessTerminating) texture->Release();
        }
    };

    // Per-string record used for selective invalidation after a tile eviction.
    struct StringInfo {
        uint32_t seq = 0;      // eviction sequence number the string's UVs were last known valid at
        uint32_t mask = 0;     // bitmask of tiles the string's glyphs live in
        bool complete = false; // false while some glyphs are still missing (string is being built incrementally)
    };

    // --- background glyph generation (runtime fonts only) ---
    struct GenJob {
        uint32_t codepoint = 0;
        int sdfW = 0, sdfH = 0;
        FT_Int bitmapLeft = 0, bitmapTop = 0;
    };
    struct GenResult {
        GenJob job;
        bool ok = false;
        bool stored = false; // already handed to the disk cache
        double ms = 0.0;
        std::vector<uint8_t> pixels;
    };
    struct GenState {
        std::mutex mtx;
        std::condition_variable cv;
        std::deque<GenJob> jobs;
        std::vector<GenResult> results;
        std::atomic<bool> hasResults{ false };
        bool stop = false;
        const FT_Byte* data = nullptr;
        FT_Long size = 0;
    };

public:
    MSDFFont(FT_Face face, const FT_Byte* fontData, FT_Long dataSize);
    MSDFFont(FT_Face face, const FT_Byte* fontData, FT_Long dataSize, FT_Long faceIndex);
    ~MSDFFont();

    bool IsValid() const { return m_isValid; }

    AtlasPage* GetAtlasPage(size_t index) const;
    size_t GetAtlasPageCount() const { return m_atlasPages.size(); }
    size_t GetAtlasEvictionCount() const { return m_evictionCount; }

    // preferSync=true: generate on the render thread (used for 3D nameplates so text does not wait on workers).
    const GlyphMetrics* GetGlyph(uint32_t codepoint, bool preferSync = false);

    static MSDFFont* Get(FT_Face face);
    static void Register(FT_Face face, const FT_Byte* data, FT_Long size);
    static void Unregister(FT_Face face);
    static void ClearAllCache();
    static void Shutdown();

    // Call at the start of each font batch: runs atlas evictions deferred from the previous batch.
    static void FlushPendingEvictions();
    static void StepDiskWarmup();

    // True if atlas pages are created with auto-generated mipmaps.
    static bool IsAutoMipEnabled() { return s_autoMip; }

    // --- selective invalidation (see MSDFAtlasTiles.h) ---
    enum class StringState { Fresh, Valid, Stale };
    // Called every frame for each string: Stale means a tile it uses was evicted and it must be rebuilt.
    StringState CheckString(const void* key, bool processedBefore);
    // Called at the start of (re)processing a string's geometry.
    void BeginString(const void* key);
    // Called at the end of a processing pass with the tiles used by the glyphs processed in it.
    void AccumulateString(const void* key, uint32_t tileMask, bool complete);

private:
    bool CreateAtlasPage();
    // Find room for a w x h glyph (may open a tile / create a page). Never evicts.
    AtlasTiles::Result AllocateSlot(uint16_t w, uint16_t h, AtlasTiles::Slot& slot);
    // Run a deferred eviction: free the least-recently-used tile(s).
    void ExecutePendingEviction();
    void EvictTile(uint32_t tile);
    bool UploadGlyphToAtlas(GlyphMetrics& metrics, uint32_t codepoint);
    bool GenerateMSDF(std::vector<uint8_t>& outData, uint32_t codepoint, int sdfW, int sdfH) const;
    // Thread-safe core of GenerateMSDF: only touches the given font handle and thread-local buffers.
    static bool GenerateMSDFWith(msdfgen::FontHandle* font, std::vector<uint8_t>& outData, uint32_t codepoint, int sdfW, int sdfH);

    // Background generation: cache misses are generated on worker threads instead of the render thread.
    void EnableAsyncGeneration() { m_asyncGen = true; }
    void StartGenWorkers();
    void StopGenWorkers();
    void DrainGenResults();
    static void GenWorkerMain(std::shared_ptr<GenState> state);

    static msdfgen::FontHandle* CreateMSDFHandle(const FT_Byte* data, FT_Long size);

    FT_Face m_ftFace;
    const FT_Byte* m_fontData = nullptr;
    FT_Long m_fontDataSize = 0;
    bool m_asyncGen = false;
    std::shared_ptr<GenState> m_gen;
    std::vector<std::thread> m_genThreads;
    ankerl::unordered_dense::map<uint32_t, GenResult> m_done; // finished by a worker, waiting for an atlas upload (render thread only)
    std::unordered_set<uint32_t> m_inflight;                  // queued or being generated (render thread only)
    msdfgen::FontHandle* m_msdfFont;
    bool m_isValid;
	uint32_t m_evictionCount;
    bool m_evictionPending = false; // atlas full: eviction is deferred to the start of the next batch

    std::unique_ptr<MSDFCache> m_cache;
    size_t m_warmupIndex = 0;
    bool m_warmupDone = false;
    std::vector<std::unique_ptr<AtlasPage>> m_atlasPages;

    ankerl::unordered_dense::map<uint32_t, GlyphMetrics> m_glyphPool;

    AtlasTiles m_tiles{ MSDF::ATLAS_SIZE, MSDF::ATLAS_GUTTER, MSDF::MAX_ATLAS_PAGES };
    EvictionLog m_evictLog;
    uint64_t m_clock = 0;
    ankerl::unordered_dense::map<const void*, StringInfo> m_strings;

    inline static ankerl::unordered_dense::map<FT_Face, std::unique_ptr<MSDFFont>> s_fontHandles;

    inline static thread_local VectorPool<float> m_msdfPool;

    // Whether atlas pages use auto-generated mipmaps (falls back to false if creation fails)
    inline static bool s_autoMip = true;
};
