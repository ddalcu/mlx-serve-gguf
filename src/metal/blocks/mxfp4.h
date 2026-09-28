// MXFP4: an E8M0 scale byte (2^(e-127), halved like ggml) and 16 bytes of FP4 nibbles.
constant int8_t kvalues_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
template <typename O>
inline void deq_mxfp4(const device uint8_t *b, uint, thread O &o) {
    const uint e = b[0];
    const float d = as_type<float>(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
    for (uint i = 0; i < 4; i++) {
        const uint4 q = GG_U4(b + 1 + 4 * i);
        const uint4 lo = q & 0xF;
        const uint4 hi = q >> 4;
        o.emit(i, d * float4(kvalues_fp4[lo.x], kvalues_fp4[lo.y], kvalues_fp4[lo.z], kvalues_fp4[lo.w]));
        o.emit(i + 4, d * float4(kvalues_fp4[hi.x], kvalues_fp4[hi.y], kvalues_fp4[hi.z], kvalues_fp4[hi.w]));
    }
}
