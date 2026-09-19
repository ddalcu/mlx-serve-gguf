// One decoder per ggml block format: deq_<type>(block, u, o) decodes UNIT u of
// a block and hands every group of 4 weights to the emitter: o.emit(i, v) is
// weights [4i, 4i+4) of that unit. A unit is the smallest run of weights that
// shares loaded bytes, so each byte is read once: a whole 32-weight block
// (Q8_0, IQ4_NL), a 32-weight ib32 group of the IQ grids, 64 weights for
// Q4_K/Q5_K (both nibbles of 32 bytes), 128 for Q2_K/Q3_K/Q6_K (all bit
// planes of 32 bytes). Units are what a matvec stripes across GPU threads.
// A matvec multiplies and accumulates on the fly (GgDot, against up to 8
// activation rows at once), a dequant writes
// straight to the output (GgOut). Neither keeps a scratch array: 32 floats of
// thread memory already spill out of registers and cost ~10x.
// Same math as src/quants.zig, which is checked against ggml.
// Loads go through packed (alignment 1) vector types, four bytes at a time,
// since blocks sit at arbitrary byte offsets.

template <typename T>
struct GgOut {
    device T *out;
    void emit(uint i, float4 v) {
        out[4 * i] = T(v.x);
        out[4 * i + 1] = T(v.y);
        out[4 * i + 2] = T(v.z);
        out[4 * i + 3] = T(v.w);
    }
};

// One unit against M activation rows of K values each (`x` = the unit's first
// value in row 0): a verify step or a decode batch pays for the weights once.
// M is a constant so the loop unrolls into M vector registers.
template <typename T, uint M, uint K>
struct GgDot {
    const device T *x;
    float4 acc[M];
    void emit(uint i, float4 v) {
        for (uint m = 0; m < M; m++) {
            const device T *xm = x + m * K + 4 * i;
            acc[m] += v * float4(xm[0], xm[1], xm[2], xm[3]);
        }
    }
};

// Weights of one row into a [column][row] tile of 8 rows, the layout an 8x8
// simdgroup_matrix load reads column block by column block.
struct GgTile {
    threadgroup float *tile;
    uint row;
    void emit(uint i, float4 v) {
        tile[(4 * i) * 8 + row] = v.x;
        tile[(4 * i + 1) * 8 + row] = v.y;
        tile[(4 * i + 2) * 8 + row] = v.z;
        tile[(4 * i + 3) * 8 + row] = v.w;
    }
};

#define GG_U4(p) uint4(uchar4(*(const device packed_uchar4 *)(p)))
#define GG_I4(p) int4(char4(*(const device packed_char4 *)(p)))

inline float gg_half(const device uint8_t *b, uint off) {
    return float(as_type<half>(uint16_t(uint16_t(b[off]) | (uint16_t(b[off + 1]) << 8))));
}

inline uint32_t gg_u32(const device uint8_t *b, uint off) {
    return uint32_t(b[off]) | (uint32_t(b[off + 1]) << 8) | (uint32_t(b[off + 2]) << 16) | (uint32_t(b[off + 3]) << 24);
}

// +-1 per weight from a sign byte, bit j set = weight j negative. `hi` picks bits 4..7.
inline float4 gg_signs(uint bits, bool hi) {
    const uint4 mask = hi ? uint4(16, 32, 64, 128) : uint4(1, 2, 4, 8);
    return select(float4(1.0f), float4(-1.0f), (uint4(bits) & mask) != 0);
}

// The 4 magnitudes packed in one 32-bit grid word, low byte first.
inline float4 gg_grid4(uint32_t word) {
    return float4(uint4(word, word >> 8, word >> 16, word >> 24) & 0xFF);
}

template <typename O>
inline void deq_q8_0(const device uint8_t *b, uint, thread O &o) {
    const float d = gg_half(b, 0);
    for (uint i = 0; i < 8; i++) o.emit(i, d * float4(GG_I4(b + 2 + 4 * i)));
}

// 32 weights from 16 bytes of codebook nibbles, shared by IQ4_NL and IQ4_XS.
template <typename O>
inline void gg_iq4_nibbles(const device uint8_t *qs, float d, thread O &o) {
    for (uint i = 0; i < 4; i++) {
        const uint4 q = GG_U4(qs + 4 * i);
        const uint4 lo = q & 0xF;
        const uint4 hi = q >> 4;
        o.emit(i, d * float4(kvalues_iq4nl[lo.x], kvalues_iq4nl[lo.y], kvalues_iq4nl[lo.z], kvalues_iq4nl[lo.w]));
        o.emit(i + 4, d * float4(kvalues_iq4nl[hi.x], kvalues_iq4nl[hi.y], kvalues_iq4nl[hi.z], kvalues_iq4nl[hi.w]));
    }
}

template <typename O>
inline void deq_iq4_nl(const device uint8_t *b, uint, thread O &o) { gg_iq4_nibbles(b + 2, gg_half(b, 0), o); }

// 6-bit scale of unit ib: low nibble in scales_l (b + 4), top 2 bits in scales_h (b + 2).
template <typename O>
inline void deq_iq4_xs(const device uint8_t *b, uint ib, thread O &o) {
    const uint scales_h = uint(b[2]) | (uint(b[3]) << 8);
    const uint ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    gg_iq4_nibbles(b + 8 + 16 * ib, gg_half(b, 0) * float(int(ls) - 32), o);
}

// Q2_K / Q3_K, unit n: weight (j, l) = bits 2j of q[32n + l], l < 32, scale group 8n + 2j + l/16.
template <typename O>
inline void deq_q2_k(const device uint8_t *b, uint n, thread O &o) {
    const float d = gg_half(b, 80);
    const float mn = gg_half(b, 82);
    for (uint h = 0; h < 2; h++) {
        for (uint c = 0; c < 4; c++) {
            const uint4 q = GG_U4(b + 16 + 32 * n + 16 * h + 4 * c);
            for (uint j = 0; j < 4; j++) {
                const uint8_t s8 = b[8 * n + 2 * j + h];
                o.emit(8 * j + 4 * h + c, d * float(s8 & 0xF) * float4((q >> (2 * j)) & 3) - mn * float(s8 >> 4));
            }
        }
    }
}

template <typename O>
inline void deq_q3_k(const device uint8_t *b, uint n, thread O &o) {
    const float d_all = gg_half(b, 108);
    const device uint8_t *scales = b + 96;
    for (uint h = 0; h < 2; h++) {
        float dl[4];
        for (uint j = 0; j < 4; j++) {
            // 6-bit scale `is`: low nibbles live in scales[0..8), the high 2 bits in scales[8..12).
            const uint is = 8 * n + 2 * j + h;
            const uint low = is < 8 ? (scales[is] & 0xF) : (scales[is - 8] >> 4);
            const uint high = (scales[8 + is % 4] >> (2 * (is / 4))) & 3;
            dl[j] = d_all * float(int(low | (high << 4)) - 32);
        }
        for (uint c = 0; c < 4; c++) {
            const uint4 q = GG_U4(b + 32 + 32 * n + 16 * h + 4 * c);
            const uint4 hm = GG_U4(b + 16 * h + 4 * c);
            for (uint j = 0; j < 4; j++) {
                const int4 lo = int4((q >> (2 * j)) & 3);
                // hmask bit set = no -4 offset.
                const int4 hi = int4(((hm >> (4 * n + j)) & 1) ^ 1) * 4;
                o.emit(8 * j + 4 * h + c, dl[j] * float4(lo - hi));
            }
        }
    }
}

inline void gg_scale_min_k4(uint j, const device uint8_t *q, thread float &sc, thread float &mn) {
    if (j < 4) {
        sc = float(q[j] & 63);
        mn = float(q[j + 4] & 63);
    } else {
        sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Q4_K and Q5_K share the layout, Q5_K adds a high-bit plane (qh, bit 2j+h of
// qh[l]). Unit j: weight (h, l) = nibble h of qs[32j + l].
template <typename O>
inline void deq_q45_k(const device uint8_t *b, uint j, thread O &o, bool five) {
    const float d = gg_half(b, 0);
    const float mn = gg_half(b, 2);
    const device uint8_t *qs = b + (five ? 48 : 16) + 32 * j;
    float s0, m0, s1, m1;
    gg_scale_min_k4(2 * j, b + 4, s0, m0);
    gg_scale_min_k4(2 * j + 1, b + 4, s1, m1);
    for (uint c = 0; c < 8; c++) {
        const uint4 q = GG_U4(qs + 4 * c);
        uint4 lo = q & 0xF;
        uint4 hi = q >> 4;
        if (five) {
            const uint4 qh = GG_U4(b + 16 + 4 * c) >> (2 * j);
            lo += (qh & 1) * 16;
            hi += ((qh >> 1) & 1) * 16;
        }
        o.emit(c, d * s0 * float4(lo) - mn * m0);
        o.emit(8 + c, d * s1 * float4(hi) - mn * m1);
    }
}

template <typename O>
inline void deq_q4_k(const device uint8_t *b, uint j, thread O &o) { deq_q45_k(b, j, o, false); }
template <typename O>
inline void deq_q5_k(const device uint8_t *b, uint j, thread O &o) { deq_q45_k(b, j, o, true); }

// Unit n, weight (k, l): low nibble from ql[64n + l + 32(k&1)] (high nibble for
// k >= 2), top 2 bits from bits 2k of qh[32n + l], scale sc[8n + 2k + l/16].
template <typename O>
inline void deq_q6_k(const device uint8_t *b, uint n, thread O &o) {
    const float d = gg_half(b, 208);
    const device uint8_t *ql = b + 64 * n;
    const device uint8_t *qh = b + 128 + 32 * n;
    const device uint8_t *sc = b + 192 + 8 * n;
    // Two passes (k = 0,2 off ql[l], then k = 1,3 off ql[l + 32]) keep the live state small.
    for (uint p = 0; p < 2; p++) {
        for (uint g = 0; g < 2; g++) {
            const float dlo = d * float(int8_t(sc[g + 2 * p]));
            const float dhi = d * float(int8_t(sc[g + 2 * p + 4]));
            for (uint c = 4 * g; c < 4 * g + 4; c++) {
                const uint4 q = GG_U4(ql + 32 * p + 4 * c);
                const uint4 h = GG_U4(qh + 4 * c) >> (2 * p);
                o.emit(8 * p + c, dlo * float4((q & 0xF) | ((h & 3) << 4)) - 32.0f * dlo);
                o.emit(8 * p + 16 + c, dhi * float4((q >> 4) | (((h >> 4) & 3) << 4)) - 32.0f * dhi);
            }
        }
    }
}

// The IQ grids, unit = ggml's ib32: 8 weights per grid entry (iq2) or 4 (iq3).
template <typename O>
inline void deq_iq2_xxs(const device uint8_t *b, uint ib, thread O &o) {
    const device uint8_t *aux8 = b + 2 + 8 * ib;
    const uint32_t aux1 = gg_u32(aux8, 4);
    const float db = gg_half(b, 0) * (0.5f + float(aux1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; l++) {
        const uint64_t grid = iq2xxs_grid[aux8[l]];
        const uint signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
        o.emit(2 * l, db * gg_grid4(uint32_t(grid)) * gg_signs(signs, false));
        o.emit(2 * l + 1, db * gg_grid4(uint32_t(grid >> 32)) * gg_signs(signs, true));
    }
}

template <typename O>
inline void deq_iq2_s(const device uint8_t *b, uint ib, thread O &o) {
    const float d = gg_half(b, 0);
    const device uint8_t *qs = b + 2 + 4 * ib;
    const device uint8_t *signs = b + 34 + 4 * ib;
    const uint qh = b[66 + ib];
    const uint8_t scales = b[74 + ib];
    const float db[2] = {
        d * (0.5f + float(scales & 0xF)) * 0.25f,
        d * (0.5f + float(scales >> 4)) * 0.25f,
    };
    for (uint l = 0; l < 4; l++) {
        const uint64_t grid = iq2s_grid[uint(qs[l]) | ((qh << (8 - 2 * l)) & 0x300)];
        o.emit(2 * l, db[l / 2] * gg_grid4(uint32_t(grid)) * gg_signs(signs[l], false));
        o.emit(2 * l + 1, db[l / 2] * gg_grid4(uint32_t(grid >> 32)) * gg_signs(signs[l], true));
    }
}

template <typename O>
inline void deq_iq3_xxs(const device uint8_t *b, uint ib, thread O &o) {
    const device uint8_t *qs = b + 2 + 8 * ib;
    const uint32_t aux = gg_u32(b, 66 + 4 * ib);
    const float db = gg_half(b, 0) * (0.5f + float(aux >> 28)) * 0.5f;
    for (uint l = 0; l < 4; l++) {
        const uint signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        o.emit(2 * l, db * gg_grid4(iq3xxs_grid[qs[2 * l]]) * gg_signs(signs, false));
        o.emit(2 * l + 1, db * gg_grid4(iq3xxs_grid[qs[2 * l + 1]]) * gg_signs(signs, true));
    }
}

template <typename O>
inline void deq_iq3_s(const device uint8_t *b, uint ib, thread O &o) {
    const device uint8_t *qs = b + 2 + 8 * ib;
    const uint qh = b[66 + ib];
    const device uint8_t *signs = b + 74 + 4 * ib;
    const uint8_t scales = b[106 + ib / 2];
    const uint nib = (ib % 2 == 0) ? (scales & 0xF) : (scales >> 4);
    const float db = gg_half(b, 0) * float(1 + 2 * nib);
    for (uint l = 0; l < 4; l++) {
        o.emit(2 * l, db * gg_grid4(iq3s_grid[uint(qs[2 * l]) | ((qh << (8 - 2 * l)) & 256)]) * gg_signs(signs[l], false));
        o.emit(2 * l + 1, db * gg_grid4(iq3s_grid[uint(qs[2 * l + 1]) | ((qh << (7 - 2 * l)) & 256)]) * gg_signs(signs[l], true));
    }
}

template <typename O>
inline void deq_iq2_xs(const device uint8_t *b, uint ib, thread O &o) {
    const float d = gg_half(b, 0);
    const device uint8_t *qs = b + 2 + 8 * ib;
    const uint8_t scales = b[66 + ib];
    const float db[2] = {
        d * (0.5f + float(scales & 0xF)) * 0.25f,
        d * (0.5f + float(scales >> 4)) * 0.25f,
    };
    for (uint l = 0; l < 4; l++) {
        const uint q = uint(qs[2 * l]) | (uint(qs[2 * l + 1]) << 8);
        const uint64_t grid = iq2xs_grid[q & 511];
        const uint signs = ksigns_iq2xs[q >> 9];
        o.emit(2 * l, db[l / 2] * gg_grid4(uint32_t(grid)) * gg_signs(signs, false));
        o.emit(2 * l + 1, db[l / 2] * gg_grid4(uint32_t(grid >> 32)) * gg_signs(signs, true));
    }
}

// IQ1: 8 ternary weights per grid entry (byte j = weights j and j + 4, a nibble
// each, value + 1), shifted by +-1/8 per group. `base` = dl * (-1 +- 1/8).
template <typename O>
inline void gg_iq1_group(uint idx, uint l, float dl, float base, thread O &o) {
    const uint32_t grid = iq1s_grid_gpu[idx];
    const uint4 q = uint4(grid, grid >> 8, grid >> 16, grid >> 24);
    o.emit(2 * l, dl * float4(q & 0xF) + base);
    o.emit(2 * l + 1, dl * float4((q >> 4) & 0xF) + base);
}

template <typename O>
inline void deq_iq1_s(const device uint8_t *b, uint ib, thread O &o) {
    const device uint8_t *qs = b + 2 + 4 * ib;
    const uint qh = uint(b[34 + 2 * ib]) | (uint(b[35 + 2 * ib]) << 8);
    const float dl = gg_half(b, 0) * float(2 * ((qh >> 12) & 7) + 1);
    const float base = dl * ((qh & 0x8000) ? -1.125f : -0.875f);
    for (uint l = 0; l < 4; l++) gg_iq1_group(uint(qs[l]) | (((qh >> (3 * l)) & 7) << 8), l, dl, base, o);
}

template <typename O>
inline void deq_iq1_m(const device uint8_t *b, uint ib, thread O &o) {
    const device uint8_t *sc = b + 48;
    // The block's f16 scale is the top nibble of each of the 4 u16 scale words.
    const uint16_t d_bits = uint16_t((sc[1] >> 4) | (sc[3] & 0xF0) | (uint(sc[5] >> 4) << 8) | (uint(sc[7] & 0xF0) << 8));
    const float d = float(as_type<half>(d_bits));
    const uint s16 = (uint(sc[2 * (ib / 2)]) | (uint(sc[2 * (ib / 2) + 1]) << 8)) >> (6 * (ib % 2));
    for (uint l = 0; l < 4; l++) {
        const float dl = d * float(2 * ((s16 >> (3 * (l / 2))) & 7) + 1);
        const uint qh = b[32 + 2 * ib + l / 2] >> (4 * (l % 2));
        gg_iq1_group(uint(b[4 * ib + l]) | ((qh & 7) << 8), l, dl, dl * ((qh & 8) ? -1.125f : -0.875f), o);
    }
}
