#pragma once

inline auto* vertexShaderHLSL = R"(
	uniform float4x4 WorldViewProj;
	float4 control : register(c23); // font size, outline mode, spread, atlas size

	struct VS_IN {
		float4 pos  : POSITION0;
		float4 col  : COLOR0;
		float2 uv0  : TEXCOORD0;
	};

	struct VS_OUT {
		float4 hpos : POSITION;
		float4 col  : COLOR0;
		float2 uv0  : TEXCOORD0;
		float4 pageIdx : TEXCOORD1; // target page index
	};

	VS_OUT main(VS_IN IN) {
		VS_OUT OUT;
		OUT.hpos = mul(IN.pos, WorldViewProj);
		OUT.col  = IN.col;

		if (control.x < 1.0f) {
			OUT.uv0  = IN.uv0;
			OUT.pageIdx = float4(0, 0, 0, 1.0f);
		} else {
			// encoded sign bits
			float bit0 = (IN.uv0.x < 0.0f) ? 1.0f : 0.0f;
			float bit1 = (IN.uv0.y < 0.0f) ? 2.0f : 0.0f;
			OUT.pageIdx = float4(bit0 + bit1, 0, 0, 0);
			OUT.uv0 = abs(IN.uv0);
		}
		return OUT;
	}
)";

inline auto* pixelShaderHLSL = R"(
	sampler2D gameTexture : register(s0);

	sampler2D sdfAtlas0   : register(s12);
	sampler2D sdfAtlas1   : register(s13);
	sampler2D sdfAtlas2   : register(s14);
	sampler2D sdfAtlas3   : register(s15);

	float4 control : register(c23); // font size, outline mode, spread, atlas size

	struct PS_IN {
		float4 col : COLOR0;
		float2 uv0 : TEXCOORD0;
		float4 pageIdx : TEXCOORD1; // target page index
	};

	float median(float r, float g, float b) {
		return max(min(r, g), min(max(r, g), b));
	}

	float4 main(PS_IN IN) : COLOR {
		if (control.x < 1.0f) return tex2D(gameTexture, IN.uv0) * IN.col;

		float2 uv = IN.uv0;

		// control.y = outline mode (0/1/2) + 10 * debug bits (3D text only, see MSDF.cpp hotkeys):
		//   bit0 = paint the whole quad magenta, bit1 = "safe AA" (mip 0 + screenPxRange floor of 1)
		float dbg = floor(control.y / 10.0f + 0.01f);
		float outlineHint = control.y - dbg * 10.0f;
		bool dbgBoxes = (dbg - 2.0f * floor(dbg / 2.0f)) > 0.5f;
		bool dbgSafe = dbg > 1.5f;
		float fontSize = control.x;
		float outlinePx = 0.0f;

		if (outlineHint >= 1.5f) {
			outlinePx = max(2.75f, pow(fontSize, 0.150f) * 1.6f);
		} else if (outlineHint >= 0.5f) {
			outlinePx = max(1.50f, pow(fontSize, 0.075f) * 1.5f);
		}

		int atlasPage = int(IN.pageIdx.x + 0.5f);

		// atlas texels covered per screen pixel (>1 means minified)
		float footprint = max(fwidth(uv.x), fwidth(uv.y)) * control.a;
		// use mips when minified to avoid shimmer; LOD capped at 2 (gutter is only ~10px, higher levels bleed into neighbours)
		float lod = clamp(log2(max(footprint, 1.0f)) - 0.5f, 0.0f, 2.0f);
		if (dbgSafe) lod = 0.0f;
		float4 uvl = float4(uv, 0.0f, lod);

		float4 sample;
		if (atlasPage == 0) sample = tex2Dlod(sdfAtlas0, uvl);
		else if (atlasPage == 1) sample = tex2Dlod(sdfAtlas1, uvl);
		else if (atlasPage == 2) sample = tex2Dlod(sdfAtlas2, uvl);
		else sample = tex2Dlod(sdfAtlas3, uvl);

		float sd = median(sample.r, sample.g, sample.b);
		float screenPxRange = (control.z / max(footprint, 1e-6)) * (1.0f - min(0.3f, fontSize * 0.0035f)); // smoother edges for larger text
		if (dbgSafe) screenPxRange = max(screenPxRange, 1.0f);
		float opacity = saturate((sd - 0.5f) * screenPxRange + 0.5f);

		if (dbgBoxes) {
			// DIAG: no clip -> the whole padded quad is visible; the glyph keeps its own colour on top.
			return float4(lerp(float3(1.0f, 0.0f, 1.0f), IN.col.rgb, opacity), max(opacity, 0.45f));
		}

		float4 result;
		if (outlinePx > 0.0f) {
			// *5 of the outline msdf, mult has to match alpha-channel range
			result = float4(
				lerp(float3(0.0f, 0.0f, 0.0f), IN.col.rgb, opacity),
				max(opacity, saturate((sample.a - 0.5f) * screenPxRange * 5.0f + outlinePx)) * IN.col.a
			);
		} else {
			result = float4(IN.col.rgb, opacity * IN.col.a);
		}

		// Each glyph quad is padded by the SDF spread, and neighbouring quads overlap in that padding.
		// Pixels that end up fully transparent must not be written at all: with depth testing/writing on
		// (3D text such as names above heads) they would otherwise hide the next glyph's edge,
		// and which pixels lose depends on the camera angle.
		clip(result.a - 0.004f);
		return result;
	}
)";