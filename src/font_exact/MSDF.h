#pragma once
#include <windows.h>
#include <filesystem>
#undef min
#undef max

#include "GameClient.h"
#include "D3D.h"

#include <msdfgen.h>
#include <msdfgen-ext.h>

struct GlyphMetrics {
    uint16_t width = 0;
    uint16_t height = 0;
	FT_Int bitmapTop = 0;
	FT_Int bitmapLeft = 0;
    float u0 = 0.0f, v0 = 0.0f, u1 = 0.0f, v1 = 0.0f;
    uint16_t atlasPageIndex = 0;
    uint8_t atlasTile = 0xFF; // tile inside the atlas (0xFF = none, e.g. blank glyphs)
    // false = the pixels are a plain SDF (fast to make); true = a real MSDF (or unknown, e.g. made by an older build).
    // A glyph drawn larger than MSDF_ABOVE_SCALE x SDF_RENDER_SIZE is refined from SDF to MSDF in the background.
    bool msdf = true;
    const uint8_t* pixelData = nullptr;
};

struct GlyphMetricsToStore {
    uint32_t codepoint = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    FT_Int bitmapTop = 0;
	FT_Int bitmapLeft = 0;
    std::vector<uint8_t> ownedPixelData;
    uint32_t dataSize = 0;
    bool msdf = true; // false = plain SDF (stored with a flag in the disk cache)
};

class MSDFCache;
class MSDFFont;

namespace MSDF {
	// ----  if you want overkill quality, try raising these
	inline constexpr uint32_t ATLAS_SIZE = 2048; // 1024-2048
	inline constexpr uint32_t PREGEN_START_KEY = VK_F11;
	inline constexpr uint32_t SDF_SAMPLER_SLOT = 23;
	inline constexpr uint32_t ATLAS_GUTTER = 10; // usually spread + 2-4
	inline constexpr uint32_t SDF_RENDER_SIZE = 64; // 48-128 (中文客户端：降低以提高图集容量)
	inline constexpr uint32_t SDF_SPREAD = 8; // 6-12
	inline constexpr D3DFORMAT D3DFMT = D3DFMT_A8R8G8B8; // D3DFMT_A8R8G8B8-D3DFMT_A16B16G16R16
	// ----

	inline CGxDevice::ShaderData*& g_FontPixelShader = *reinterpret_cast<CGxDevice::ShaderData**>(0x00C7D2CC);
	inline CGxDevice::ShaderData*& g_FontVertexShader = *reinterpret_cast<CGxDevice::ShaderData**>(0x00C7D2D0);

	inline FT_Library g_realFtLibrary = nullptr;
	inline msdfgen::FreetypeHandle* g_msdfFreetype = nullptr;

    // A plain SDF matches an MSDF up to about the render size of the cell and rounds sharp corners beyond it.
    // 2D text drawn larger than this factor of SDF_RENDER_SIZE gets its glyphs refined to MSDF.
    inline constexpr double MSDF_ABOVE_SCALE = 1.0;

    inline constexpr uint32_t MAX_ATLAS_PAGES = 4;
    inline constexpr size_t CJK_CACHE_THRESHOLD = 16661;

	inline bool IS_CJK = false;
	inline bool INITIALIZED = false;
	inline bool ALLOW_UNSAFE_FONTS = false;

	inline const bool IS_WIN10 = []() {
		HMODULE hKernel = GetModuleHandleW(L"kernelbase.dll");
		if (!hKernel) return false;

		return (GetProcAddress(hKernel, "VirtualAlloc2") != nullptr &&
			GetProcAddress(hKernel, "MapViewOfFile3") != nullptr &&
			GetProcAddress(hKernel, "UnmapViewOfFile2") != nullptr);
		}();

	inline std::string GetGameLocale() {
		static const std::string locale = []() {
			char exePath[MAX_PATH] = {0};
			if (!GetModuleFileNameA(nullptr, exePath, MAX_PATH)) {
				return std::string{"enUS"};
			}

			std::filesystem::path gameRoot(exePath);
			gameRoot = gameRoot.parent_path();

			constexpr const char* locales[] = {
				"enUS", "enGB", "deDE", "frFR", "esES", "esMX", "ptBR", "ruRU", "zhCN", "zhTW", "koKR"
			};

			std::error_code ec;
			for (const char* candidate : locales) {
				const std::filesystem::path localeDir = gameRoot / "Data" / candidate;
				if (!std::filesystem::exists(localeDir, ec)) {
					ec.clear();
					continue;
				}

				if (std::filesystem::exists(localeDir / (std::string("locale-") + candidate + ".MPQ"), ec) ||
					std::filesystem::exists(localeDir / (std::string("patch-") + candidate + ".MPQ"), ec) ||
					std::filesystem::exists(localeDir / (std::string("base-") + candidate + ".MPQ"), ec)) {
					return std::string{candidate};
				}
				ec.clear();
			}

			return std::string{"enUS"};
		}();

		return locale;
	}

    void initialize();
};
