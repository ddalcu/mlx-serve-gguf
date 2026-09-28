template <typename O>
inline void deq_q8_0(const device uint8_t *b, uint, thread O &o) {
    const float d = gg_half(b, 0);
    for (uint i = 0; i < 8; i++) o.emit(i, d * float4(GG_I4(b + 2 + 4 * i)));
}
