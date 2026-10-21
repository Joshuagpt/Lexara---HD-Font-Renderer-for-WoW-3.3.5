#include "../Logger.h"
#include "MSDFFont.h"
#include "MSDFCache.h"
#include "MSDFValidator.h"
#include "MSDFUtils.h"
#include <ranges>

MSDFFont::MSDFFont(FT_Face face, const FT_Byte* fontData, FT_Long dataSize)
    : m_ftFace(face), m_msdfFont(nullptr), m_isValid(false), m_evictionCount(0)
{
    if (!face) return;
    m_fontData = fontData;
    m_fontDataSize = dataSize;

    m_msdfFont = CreateMSDFHandle(fontData, dataSize);
    if (!m_msdfFont) return;

    m_cache = std::make_unique<MSDFCache>(fontData, dataSize,
        face->family_name ? face->family_name : "Unknown", face->style_name ? face->style_name : "",
        MSDF::SDF_RENDER_SIZE, MSDF::SDF_SPREAD);

    m_isValid = m_cache->GetManifestSize() || MSDF::ALLOW_UNSAFE_FONTS || MSDFValidator::IsFontMSDFCompatible(m_msdfFont);
    if (m_isValid) m_glyphPool.reserve(4096);
}

MSDFFont::MSDFFont(FT_Face face, const FT_Byte* fontData, FT_Long dataSize, FT_Long)
    : MSDFFont(face, fontData, dataSize) {
}

MSDFFont::~MSDFFont() {
    StopGenWorkers();
    m_glyphPool.clear();
    m_atlasPages.clear();
    m_cache.reset();
    if (m_msdfFont) {
        msdfgen::destroyFont(m_msdfFont);
        m_msdfFont = nullptr;
    }
}

MSDFFont* MSDFFont::Get(FT_Face face) {
    auto it = s_fontHandles.find(face);
    if (it != s_fontHandles.end() && it->second->IsValid()) {
        return it->second.get();
    }
    return nullptr;
}

void MSDFFont::Register(FT_Face face, const FT_Byte* data, FT_Long size) {
    if (s_fontHandles.find(face) != s_fontHandles.end()) return;
    auto font = std::make_unique<MSDFFont>(face, data, size);
    if (font->m_msdfFont && font->m_isValid) {
        // Runtime fonts write their disk cache on a background thread (pre-generation stays synchronous).
        font->m_cache->SetAsyncWrites(true);
        // Cache misses are generated on worker threads (pre-generation keeps the synchronous path).
        font->EnableAsyncGeneration();
        s_fontHandles[face] = std::move(font);
    }
}

void MSDFFont::Unregister(FT_Face face) {
    auto it = s_fontHandles.find(face);
    if (it != s_fontHandles.end()) {
        s_fontHandles.erase(it);
    }
}

void MSDFFont::ClearAllCache() {
    for (auto& handle : s_fontHandles | std::views::values) {
        if (handle) {
            handle->m_glyphPool.clear();
            handle->m_atlasPages.clear();
            handle->m_tiles.Reset();
            handle->m_strings.clear();
            handle->m_evictLog.Record(0xFFFFFFFFu); // everything is invalid
            handle->m_evictionPending = false;
            handle->m_evictionCount++;
        }
    }
}

void MSDFFont::Shutdown() {
    s_fontHandles.clear();
}

void MSDFFont::FlushPendingEvictions() {
    for (auto& handle : s_fontHandles | std::views::values) {
        if (!handle) continue;
        handle->m_tiles.SetClock(++handle->m_clock);
        if (handle->m_evictionPending) handle->ExecutePendingEviction();
    }
}

void MSDFFont::StepDiskWarmup() {
    constexpr size_t kPerFontPerFrame = 16;
    for (auto& handle : s_fontHandles | std::views::values) {
        if (!handle || !handle->m_cache || handle->m_warmupDone || !handle->m_isValid) continue;

        std::vector<uint32_t> batch;
        batch.reserve(kPerFontPerFrame);
        const size_t n = handle->m_cache->CollectCachedCodepoints(batch, kPerFontPerFrame, handle->m_warmupIndex);
        if (n == 0) {
            handle->m_warmupDone = true;
            continue;
        }
        for (uint32_t cp : batch) {
            if (handle->m_glyphPool.find(cp) != handle->m_glyphPool.end()) continue;
            // Disk only — GetGlyph will TryLoadGlyph then upload (no worker for cache hits)
            handle->GetGlyph(cp, false);
        }
        if (handle->m_evictionPending) break;
    }
}

void MSDFFont::ExecutePendingEviction() {
    m_evictionPending = false;
    // Free at most a couple of tiles per batch; glyphs that did not fit are retried on later frames.
    constexpr int kMaxTilesPerBatch = 2;
    for (int n = 0; n < kMaxTilesPerBatch; ++n) {
        int victim = m_tiles.PickVictim();
        if (victim < 0) break;
        EvictTile(static_cast<uint32_t>(victim));
    }
}

void MSDFFont::EvictTile(uint32_t tile) {
    for (uint32_t cp : m_tiles.Evict(tile)) {
        m_glyphPool.erase(cp);
    }

    const uint32_t page = tile / AtlasTiles::TILES_PER_PAGE;
    if (page < m_atlasPages.size() && m_atlasPages[page]->texture) {
        const uint32_t ox = m_tiles.TileOriginX(tile), oy = m_tiles.TileOriginY(tile), sz = m_tiles.TileSize();
        RECT r = { static_cast<LONG>(ox), static_cast<LONG>(oy), static_cast<LONG>(ox + sz), static_cast<LONG>(oy + sz) };
        D3DLOCKED_RECT lr;
        if (SUCCEEDED(m_atlasPages[page]->texture->LockRect(0, &lr, &r, 0))) {
            auto* row = static_cast<unsigned char*>(lr.pBits);
            for (uint32_t y = 0; y < sz; ++y, row += lr.Pitch) memset(row, 0, static_cast<size_t>(sz) * 4);
            m_atlasPages[page]->texture->UnlockRect(0);
        }
    }

    m_evictLog.Record(1u << tile);
    m_evictionCount++;

    // Diagnostics: frequent evictions mean the working set does not fit the atlas.
    if (m_evictionCount <= 10 || (m_evictionCount % 100) == 0) {
        Log("[MSDF] atlas tile eviction #%zu: tile %u freed", m_evictionCount, tile);
    }
}

MSDFFont::StringState MSDFFont::CheckString(const void* key, bool processedBefore) {
    auto it = m_strings.find(key);
    if (it == m_strings.end()) return processedBefore ? StringState::Stale : StringState::Fresh;

    StringInfo& si = it->second;
    // A complete record on a string that has not been processed yet is a leftover from a dead
    // string that used the same address: forget it.
    if (!processedBefore && si.complete) {
        m_strings.erase(it);
        return StringState::Fresh;
    }
    if (!m_evictLog.StillValid(si.seq, si.mask)) {
        m_strings.erase(it);
        return StringState::Stale;
    }
    si.seq = m_evictLog.Seq();
    for (uint32_t m = si.mask; m; m &= m - 1) {
        m_tiles.Touch(static_cast<uint32_t>(std::countr_zero(m)));
    }
    return StringState::Valid;
}

void MSDFFont::BeginString(const void* key) {
    auto it = m_strings.find(key);
    if (it != m_strings.end() && it->second.complete) m_strings.erase(it);
}

void MSDFFont::AccumulateString(const void* key, uint32_t tileMask, bool complete) {
    // Addresses of dead strings are never reported to us; bound the table and let strings rebuild if it overflows.
    if (m_strings.size() > 16384) m_strings.clear();

    StringInfo& si = m_strings[key];
    si.mask |= tileMask;
    si.seq = m_evictLog.Seq();
    si.complete = complete;
}

const GlyphMetrics* MSDFFont::GetGlyph(uint32_t codepoint, bool preferSync) {
    auto pit = m_glyphPool.find(codepoint);
    if (pit != m_glyphPool.end()) {
        if (pit->second.atlasTile != 0xFF) m_tiles.Touch(pit->second.atlasTile);
        return &pit->second;
    }

    auto [it, inserted] = m_glyphPool.try_emplace(codepoint);
    GlyphMetrics& metrics = it->second;

    if (m_cache->TryLoadGlyph(codepoint, metrics)) {
        if (!UploadGlyphToAtlas(metrics, codepoint)) {
            // No room right now (eviction deferred) or upload failed: callers retry on later frames
            static int s_n = 0;
            if (s_n < 200 || preferSync) {
                ++s_n;
                Log("[DIAG] cache-hit glyph U+%04X upload failed (w=%u h=%u, evictPending=%d, preferSync=%d)",
                    codepoint, metrics.width, metrics.height, m_evictionPending ? 1 : 0, preferSync ? 1 : 0);
            }
            m_glyphPool.erase(it);
            return nullptr;
        }
        if (metrics.width == 0 && codepoint >= 0x2E80) {
            static int s_nb = 0;
            if (s_nb < 200) { ++s_nb; Log("[DIAG] cache-hit glyph U+%04X is BLANK (width=0) -> would be invisible", codepoint); }
        }
        return &metrics;
    }
    LARGE_INTEGER diagT0; QueryPerformanceCounter(&diagT0);

    GlyphMetricsToStore storage;
    storage.codepoint = codepoint;

    bool uploadFailed = false;
    bool fromWorker = false;   // pixels were produced by a worker thread
    bool alreadyStored = false;
    GenResult keep;            // copy kept if the atlas had no room for a worker result

    if (m_asyncGen) {
        DrainGenResults();
        auto dit = m_done.find(codepoint);
        if (dit != m_done.end()) {
            GenResult res = std::move(dit->second);
            m_done.erase(dit);
            fromWorker = true;
            alreadyStored = res.stored;
            storage.bitmapLeft = res.job.bitmapLeft;
            storage.bitmapTop = res.job.bitmapTop;
            if (res.ok) {
                keep.job = res.job; keep.ok = true; keep.ms = res.ms; keep.pixels = res.pixels; // small (<= ~100 KB)
                storage.width = static_cast<uint16_t>(res.job.sdfW);
                storage.height = static_cast<uint16_t>(res.job.sdfH);
                storage.ownedPixelData = std::move(res.pixels);
                storage.dataSize = static_cast<uint32_t>(storage.ownedPixelData.size());
                metrics.width = storage.width;
                metrics.height = storage.height;
                metrics.bitmapLeft = storage.bitmapLeft;
                metrics.bitmapTop = storage.bitmapTop;
                metrics.pixelData = storage.ownedPixelData.data();
                uploadFailed = !UploadGlyphToAtlas(metrics, codepoint);
                static int s_ns = 0;
                if (res.ms > 8.0 && s_ns < 100) { ++s_ns; Log("[DIAG] background glyph generation U+%04X: %.1f ms (off render thread)", codepoint, res.ms); }
            } else {
                static int s_n = 0;
                if (s_n < 200) { ++s_n; Log("[DIAG] GenerateMSDF FAILED for U+%04X (background) -> stored as blank", codepoint); }
            }
        } else if (m_inflight.count(codepoint)) {
            if (!preferSync) {
                // A worker is on it: 2D callers retry on later frames.
                m_glyphPool.erase(it);
                return nullptr;
            }
            // 3D preferSync: do not wait on the queue — generate on this thread.
            // Worker result, if any, will land in m_done and be ignored if pool already has the glyph.
            m_inflight.erase(codepoint);
        }
    }

    if (!fromWorker) {
        if (FT_Set_Pixel_Sizes(m_ftFace, MSDF::SDF_RENDER_SIZE, MSDF::SDF_RENDER_SIZE) != 0) {
            m_glyphPool.erase(it);
            return nullptr;
        }

        FT_UInt glyphIndex = FT_Get_Char_Index(m_ftFace, codepoint);
        if (FT_Load_Glyph(m_ftFace, glyphIndex, FT_LOAD_NO_BITMAP | FT_LOAD_NO_HINTING) != 0) {
            m_glyphPool.erase(it);
            return nullptr;
        }

        storage.bitmapLeft = m_ftFace->glyph->bitmap_left;
        storage.bitmapTop = m_ftFace->glyph->bitmap_top;

        const bool hasOutline = m_ftFace->glyph->format == FT_GLYPH_FORMAT_OUTLINE &&
            m_ftFace->glyph->outline.n_contours > 0;
        if (hasOutline) {
            FT_BBox bbox;
            FT_Outline_Get_BBox(&m_ftFace->glyph->outline, &bbox);

            uint16_t w = static_cast<uint16_t>(std::max(0, static_cast<int>(((bbox.xMax + 63) >> 6) - (bbox.xMin >> 6))));
            uint16_t h = static_cast<uint16_t>(std::max(0, static_cast<int>(((bbox.yMax + 63) >> 6) - (bbox.yMin >> 6))));

            if (w > 0 && h > 0) {
                uint16_t sdfW = w + 2 * MSDF::SDF_SPREAD;
                uint16_t sdfH = h + 2 * MSDF::SDF_SPREAD;

                // Check the atlas has room BEFORE generating, so no CPU is wasted on a glyph that cannot be placed.
                // Mark eviction as pending; it runs at the start of the next batch.
                AtlasTiles::Slot probe;
                const AtlasTiles::Result room = AllocateSlot(sdfW, sdfH, probe);
                if (room != AtlasTiles::Result::Ok) {
                    if (room == AtlasTiles::Result::Full) m_evictionPending = true;
                    static int s_n = 0;
                    if (s_n < 200) { ++s_n; Log("[DIAG] no atlas slot for U+%04X: %s (pages=%zu)", codepoint, room == AtlasTiles::Result::Full ? "FULL" : "PAGE_FAILED", m_atlasPages.size()); }
                    m_glyphPool.erase(it);
                    return nullptr;
                }

                // 3D nameplates: preferSync skips the worker queue so the glyph is ready this frame.
                if (m_asyncGen && !preferSync && sdfW <= 512 && sdfH <= 512) {
                    // Hand the expensive part to a worker thread; the render thread only uploads the result later.
                    StartGenWorkers();
                    if (!m_genThreads.empty()) {
                        GenJob job;
                        job.codepoint = codepoint;
                        job.sdfW = sdfW; job.sdfH = sdfH;
                        job.bitmapLeft = storage.bitmapLeft; job.bitmapTop = storage.bitmapTop;
                        {
                            std::lock_guard<std::mutex> lk(m_gen->mtx);
                            m_gen->jobs.push_back(job);
                        }
                        m_gen->cv.notify_one();
                        m_inflight.insert(codepoint);
                        m_glyphPool.erase(it);
                        return nullptr;
                    }
                }

                storage.ownedPixelData.reserve(static_cast<size_t>(sdfW) * sdfH * 4);
                if (GenerateMSDF(storage.ownedPixelData, codepoint, sdfW, sdfH)) {
                    storage.width = sdfW;
                    storage.height = sdfH;
                    storage.dataSize = static_cast<uint32_t>(storage.ownedPixelData.size());
                    metrics.width = storage.width;
                    metrics.height = storage.height;
                    metrics.bitmapLeft = storage.bitmapLeft;
                    metrics.bitmapTop = storage.bitmapTop;
                    metrics.pixelData = storage.ownedPixelData.data();
                    uploadFailed = !UploadGlyphToAtlas(metrics, codepoint);
                } else {
                    static int s_n = 0;
                    if (s_n < 200) { ++s_n; Log("[DIAG] GenerateMSDF FAILED for U+%04X (sdf %dx%d, glyphIndex=%u) -> stored as blank", codepoint, sdfW, sdfH, glyphIndex); }
                }
            }
        }
        LARGE_INTEGER t1, fq; QueryPerformanceCounter(&t1); QueryPerformanceFrequency(&fq);
        const double ms = (t1.QuadPart - diagT0.QuadPart) * 1000.0 / static_cast<double>(fq.QuadPart);
        static int s_ns = 0;
        if (ms > 8.0 && s_ns < 100) { ++s_ns; Log("[DIAG] slow glyph generation U+%04X: %.1f ms (render thread)", codepoint, ms); }
    }

    if (metrics.width == 0 && codepoint >= 0x2E80) {
        static int s_nb = 0;
        if (s_nb < 200) { ++s_nb; Log("[DIAG] generated glyph U+%04X is BLANK (width=0)", codepoint); }
    }

    // Always store into the disk cache, whether or not the upload succeeded
    if (!alreadyStored) {
        m_cache->StoreGlyph(std::move(storage));
    }

    if (uploadFailed) {
        // A worker result that found no atlas room is kept (marked as stored) so it is not generated again.
        if (fromWorker && keep.ok) {
            keep.stored = true;
            m_done[codepoint] = std::move(keep);
        }
        if (preferSync) {
            static int s_ps = 0;
            if (s_ps < 100) {
                ++s_ps;
                Log("[DIAG] preferSync U+%04X failed upload (evictPending=%d pages=%zu)",
                    codepoint, m_evictionPending ? 1 : 0, m_atlasPages.size());
            }
        }
        m_glyphPool.erase(it);
        return nullptr;
    }
    return &metrics;
}

void MSDFFont::StartGenWorkers() {
    if (m_gen) return;
    m_gen = std::make_shared<GenState>();
    m_gen->data = m_fontData;
    m_gen->size = m_fontDataSize;

    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned n = std::clamp<unsigned>(hw > 1 ? hw - 1 : 1, 1u, 3u);
    for (unsigned i = 0; i < n; ++i) {
        try {
            m_genThreads.emplace_back(&MSDFFont::GenWorkerMain, m_gen);
        } catch (...) {
            break;
        }
    }
    if (m_genThreads.empty()) Log("[DIAG] could not start glyph worker threads; generating on the render thread");
    else Log("[MSDF] glyph generation: %zu background worker thread(s) for this font", m_genThreads.size());
}

void MSDFFont::StopGenWorkers() {
    if (!m_gen) return;
    {
        std::lock_guard<std::mutex> lk(m_gen->mtx);
        m_gen->stop = true;
    }
    m_gen->cv.notify_all();
    for (std::thread& t : m_genThreads) {
        if (!t.joinable()) continue;
        // At process exit the OS may already have killed the workers: never wait for them.
        if (g_isProcessTerminating) t.detach();
        else t.join();
    }
    m_genThreads.clear();
}

void MSDFFont::DrainGenResults() {
    if (!m_gen || !m_gen->hasResults.load(std::memory_order_acquire)) return;
    std::vector<GenResult> local;
    {
        std::lock_guard<std::mutex> lk(m_gen->mtx);
        local.swap(m_gen->results);
        m_gen->hasResults.store(false, std::memory_order_relaxed);
    }
    for (GenResult& r : local) {
        m_inflight.erase(r.job.codepoint);
        const uint32_t cp = r.job.codepoint;
        m_done[cp] = std::move(r);
    }
}

void MSDFFont::GenWorkerMain(std::shared_ptr<GenState> st) {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);

    // Every worker owns its FreeType library and font face: msdfgen/FreeType objects are not shared across threads.
    msdfgen::FreetypeHandle* ft = msdfgen::initializeFreetype();
    msdfgen::FontHandle* font = ft ? msdfgen::loadFontData(ft, st->data, st->size) : nullptr;

    for (;;) {
        GenJob job;
        {
            std::unique_lock<std::mutex> lk(st->mtx);
            st->cv.wait(lk, [&] { return st->stop || !st->jobs.empty(); });
            if (st->stop) break;
            job = st->jobs.front();
            st->jobs.pop_front();
        }

        LARGE_INTEGER t0, t1, fq;
        QueryPerformanceCounter(&t0);
        GenResult res;
        res.job = job;
        res.ok = GenerateMSDFWith(font, res.pixels, job.codepoint, job.sdfW, job.sdfH);
        QueryPerformanceCounter(&t1);
        QueryPerformanceFrequency(&fq);
        res.ms = (t1.QuadPart - t0.QuadPart) * 1000.0 / static_cast<double>(fq.QuadPart);

        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->results.push_back(std::move(res));
            st->hasResults.store(true, std::memory_order_release);
        }
    }

    if (font) msdfgen::destroyFont(font);
    if (ft) msdfgen::deinitializeFreetype(ft);
}

MSDFFont::AtlasPage* MSDFFont::GetAtlasPage(size_t index) const {
    if (index < m_atlasPages.size()) {
        return m_atlasPages[index].get();
    }
    return nullptr;
}

bool MSDFFont::CreateAtlasPage() {
    auto page = std::make_unique<AtlasPage>();

    bool created = false;
    if (s_autoMip) {
        // Auto mipmaps: reduce shimmer / dropped strokes when minified (max LOD is clamped in the pixel shader)
        created = D3D::CreateTexture(&page->texture, {
            .width = MSDF::ATLAS_SIZE,
            .height = MSDF::ATLAS_SIZE,
            .levels = 0,
            .usage = D3DUSAGE_AUTOGENMIPMAP,
            .format = MSDF::D3DFMT,
            .pool = D3DPOOL_MANAGED
            });
        if (!created) s_autoMip = false; // unsupported by driver: fall back to a single level
    }
    if (!created) {
        created = D3D::CreateTexture(&page->texture, {
            .width = MSDF::ATLAS_SIZE,
            .height = MSDF::ATLAS_SIZE,
            .format = MSDF::D3DFMT,
            .pool = D3DPOOL_MANAGED
            });
    }
    if (!created) return false;

    m_atlasPages.push_back(std::move(page));
    return true;
}

AtlasTiles::Result MSDFFont::AllocateSlot(uint16_t w, uint16_t h, AtlasTiles::Slot& slot) {
    return m_tiles.Find(w, h, slot, [this](uint32_t page) {
        while (m_atlasPages.size() <= page) {
            if (!CreateAtlasPage()) return false;
        }
        return true;
    });
}

bool MSDFFont::UploadGlyphToAtlas(GlyphMetrics& metrics, uint32_t codepoint) {
    if (!metrics.pixelData || metrics.width == 0 || metrics.height == 0) return true;

    AtlasTiles::Slot slot;
    const AtlasTiles::Result room = AllocateSlot(metrics.width, metrics.height, slot);
    if (room != AtlasTiles::Result::Ok) {
        // Never evict mid-frame: strings already processed this frame would point at wiped atlas regions.
        if (room == AtlasTiles::Result::Full) m_evictionPending = true;
        return false;
    }
    AtlasPage* targetPage = m_atlasPages[slot.page].get();
    if (!targetPage->texture) return false;

    // Lock only this glyph's sub-rectangle: avoids dirtying and re-uploading the whole 16MB page
    RECT lockRect = {
        static_cast<LONG>(slot.x), static_cast<LONG>(slot.y),
        static_cast<LONG>(slot.x + metrics.width), static_cast<LONG>(slot.y + metrics.height)
    };
    D3DLOCKED_RECT lockedRect;
    if (FAILED(targetPage->texture->LockRect(0, &lockedRect, &lockRect, 0))) {
        return false;
    }
    if (lockedRect.Pitch < metrics.width * 4) {
        targetPage->texture->UnlockRect(0);
        return false;
    }

    const unsigned char* src = metrics.pixelData;
    unsigned char* dest = static_cast<unsigned char*>(lockedRect.pBits); // already points at the sub-rect's top-left
    for (uint16_t y = 0; y < metrics.height; ++y) {
        memcpy(dest, src, metrics.width * 4);
        dest += lockedRect.Pitch;
        src += metrics.width * 4;
    }
    targetPage->texture->UnlockRect(0);

    const float atlasSize = static_cast<float>(MSDF::ATLAS_SIZE);
    metrics.u0 = static_cast<float>(slot.x) / atlasSize;
    metrics.v0 = static_cast<float>(slot.y) / atlasSize;
    metrics.u1 = static_cast<float>(slot.x + metrics.width) / atlasSize;
    metrics.v1 = static_cast<float>(slot.y + metrics.height) / atlasSize;
    metrics.atlasPageIndex = static_cast<uint16_t>(slot.page);
    metrics.atlasTile = static_cast<uint8_t>(slot.tile);

    m_tiles.Commit(slot, metrics.width, metrics.height, codepoint);
    return true;
}

bool MSDFFont::GenerateMSDF(std::vector<uint8_t>& outData, uint32_t codepoint, int sdfW, int sdfH) const {
    return GenerateMSDFWith(m_msdfFont, outData, codepoint, sdfW, sdfH);
}

bool MSDFFont::GenerateMSDFWith(msdfgen::FontHandle* font, std::vector<uint8_t>& outData, uint32_t codepoint, int sdfW, int sdfH) {
    if (!font) return false;
    if (sdfW <= 0 || sdfH <= 0 || sdfW > 512 || sdfH > 512) return false;

    msdfgen::Shape shape;
    if (!msdfgen::loadGlyph(shape, font, codepoint)) return false;

    if (shape.contours.empty()) {
        outData.assign(sdfW * sdfH * 4, 0);
        return true;
    }

    msdfgen::resolveShapeGeometry(shape);
    msdfgen::edgeColoringInkTrap(shape, 3.0, 0);

    auto bounds = shape.getBounds();
    double shapeW = bounds.r - bounds.l;
    double shapeH = bounds.t - bounds.b;
    if (shapeW <= 0 || shapeH <= 0) return false;

    double usableW = static_cast<double>(sdfW) - 2.0 * MSDF::SDF_SPREAD;
    double usableH = static_cast<double>(sdfH) - 2.0 * MSDF::SDF_SPREAD;
    if (usableW <= 0 || usableH <= 0) return false;

    double scale = std::min(usableW / shapeW, usableH / shapeH);
    msdfgen::Projection projection(
        msdfgen::Vector2(scale, scale),
        msdfgen::Vector2(MSDF::SDF_SPREAD / scale - bounds.l, MSDF::SDF_SPREAD / scale - bounds.b)
    );

    auto msdfBuf = m_msdfPool.AcquireSized(sdfW * sdfH * 3);
    auto sdfBuf = m_msdfPool.AcquireSized(sdfW * sdfH);

    msdfgen::BitmapRef<float, 3> msdfBitmap(msdfBuf.data(), sdfW, sdfH);
    msdfgen::BitmapRef<float, 1> sdfBitmap(sdfBuf.data(), sdfW, sdfH);

    msdfgen::MSDFGeneratorConfig config;
    config.overlapSupport = true;

    msdfgen::Range msdfRange(MSDF::SDF_SPREAD / scale);
    msdfgen::generateMSDF(msdfBitmap, shape, projection, msdfRange, config);
    msdfgen::SDFTransformation msdfTransform(projection, msdfRange);
    msdfgen::distanceSignCorrection(msdfBitmap, shape, msdfTransform, msdfgen::FillRule::FILL_NONZERO);

    msdfgen::Range sdfRange(MSDF::SDF_SPREAD / scale * 5.0);
    msdfgen::generateSDF(sdfBitmap, shape, projection, sdfRange);
    msdfgen::SDFTransformation sdfTransform(projection, sdfRange);
    msdfgen::distanceSignCorrection(sdfBitmap, shape, sdfTransform, msdfgen::FillRule::FILL_NONZERO);

    outData.resize(sdfW * sdfH * 4);
    uint8_t* dest = outData.data();
    const float* srcMSDF = msdfBuf.data();
    const float* srcSDF = sdfBuf.data();

    for (int i = 0; i < sdfW * sdfH; ++i) {
        dest[i * 4 + 0] = static_cast<uint8_t>(std::clamp(srcMSDF[i * 3 + 0] * 255.f, 0.f, 255.f));
        dest[i * 4 + 1] = static_cast<uint8_t>(std::clamp(srcMSDF[i * 3 + 1] * 255.f, 0.f, 255.f));
        dest[i * 4 + 2] = static_cast<uint8_t>(std::clamp(srcMSDF[i * 3 + 2] * 255.f, 0.f, 255.f));
        dest[i * 4 + 3] = static_cast<uint8_t>(std::clamp(srcSDF[i] * 255.f, 0.f, 255.f));
    }
    m_msdfPool.Release(std::move(msdfBuf));
    m_msdfPool.Release(std::move(sdfBuf));

    return true;
}

msdfgen::FontHandle* MSDFFont::CreateMSDFHandle(const FT_Byte* data, FT_Long size) {
    return !MSDF::g_msdfFreetype ? nullptr : msdfgen::loadFontData(MSDF::g_msdfFreetype, data, size);
}
