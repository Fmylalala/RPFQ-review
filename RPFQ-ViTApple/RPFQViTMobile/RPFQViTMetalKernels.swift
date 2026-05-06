import Foundation

enum RPFQViTMetalKernels {
    static let simdWidth = 32
    static let rowsPerGroup = 8
    static let threadgroupSize = rowsPerGroup * simdWidth
    static let packedPerThread = 2

    static let decodeSource: String = """
        const uint SIMD_W = SIMD_WIDTH;
        const uint PPT = PACKED_PER_THREAD;
        const uint TG = TG_SIZE;

        uint local_id = thread_position_in_grid.x % TG;
        uint group_id = thread_position_in_grid.x / TG;
        uint row_in_group = local_id / SIMD_W;
        uint lane_id = local_id % SIMD_W;
        uint row = group_id * ROWS_PER_GROUP + row_in_group;

        threadgroup float shared_xRe[QK];
        threadgroup float shared_xIm[QK];

        float accRe = 0.0f;
        float accIm = 0.0f;

        const uint codeRowBase = row * PACKED_PER_ROW;
        const uint scaleRowBase = row * BLOCKS_PER_ROW;

        for (uint block = 0; block < BLOCKS_PER_ROW; ++block) {
            const uint xBlockBase = block * QK;

            for (uint i = local_id; i < QK; i += TG) {
                shared_xRe[i] = float(xRe[xBlockBase + i]);
                shared_xIm[i] = float(xIm[xBlockBase + i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row < OUT_DIM) {
                const uint codeBlockBase = codeRowBase + block * PACKED_PER_BLOCK;

                float sReXr = 0.0f, sReXi = 0.0f;
                float sImXr = 0.0f, sImXi = 0.0f;

                for (uint p = 0; p < PPT; ++p) {
                    const uint packed = lane_id + p * SIMD_W;
                    const uchar bits = codes[codeBlockBase + packed];

                    for (uint lane = 0; lane < 4; ++lane) {
                        const uint code = (uint(bits) >> (lane * 2)) & 0x3;
                        const uint xLocal = packed * 4u + lane;
                        const float xr = shared_xRe[xLocal];
                        const float xi = shared_xIm[xLocal];

                        const bool is_real = (code < 2u);
                        const bool raw_pos = bool(code & 1u);
                        const bool imag_pos = CODEBOOK_10_POS_IMAG ? !raw_pos : raw_pos;
                        const bool is_pos = is_real ? raw_pos : imag_pos;

                        const float xrs = is_pos ? xr : -xr;
                        const float xis = is_pos ? xi : -xi;

                        sReXr += is_real ? xrs : 0.0f;
                        sReXi += is_real ? xis : 0.0f;
                        sImXr += is_real ? 0.0f : xrs;
                        sImXi += is_real ? 0.0f : xis;
                    }
                }

                const float gr = float(gammaReal[scaleRowBase + block]);
                const float gi = float(gammaImag[scaleRowBase + block]);
                const float inv_gr = 1.0f / gr;
                const float inv_gi = 1.0f / gi;
                accRe += inv_gr * sReXr + inv_gi * sImXi;
                accIm += inv_gi * sImXr - inv_gr * sReXi;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < OUT_DIM) {
            accRe = simd_sum(accRe);
            accIm = simd_sum(accIm);

            if (lane_id == 0) {
                outRe[row] = half(accRe);
                outIm[row] = half(accIm);
            }
        }
    """

    static let prefillSource: String = """
        const uint SIMD_W = SIMD_WIDTH;
        const uint PPT = PACKED_PER_THREAD;
        const uint TG = TG_SIZE;

        uint local_id = thread_position_in_grid.x % TG;
        uint group_id = thread_position_in_grid.x / TG;
        uint row_in_group = local_id / SIMD_W;
        uint lane_id = local_id % SIMD_W;
        uint row = group_id * ROWS_PER_GROUP + row_in_group;
        uint tok = thread_position_in_grid.y;

        threadgroup float shared_xRe[QK];
        threadgroup float shared_xIm[QK];

        float accRe = 0.0f;
        float accIm = 0.0f;

        const uint codeRowBase = row * PACKED_PER_ROW;
        const uint scaleRowBase = row * BLOCKS_PER_ROW;
        const uint xTokBase = tok * IN_DIM;

        for (uint block = 0; block < BLOCKS_PER_ROW; ++block) {
            const uint xBlockBase = xTokBase + block * QK;

            for (uint i = local_id; i < QK; i += TG) {
                shared_xRe[i] = float(xRe[xBlockBase + i]);
                shared_xIm[i] = float(xIm[xBlockBase + i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row < OUT_DIM) {
                const uint codeBlockBase = codeRowBase + block * PACKED_PER_BLOCK;

                float sReXr = 0.0f, sReXi = 0.0f;
                float sImXr = 0.0f, sImXi = 0.0f;

                for (uint p = 0; p < PPT; ++p) {
                    const uint packed = lane_id + p * SIMD_W;
                    const uchar bits = codes[codeBlockBase + packed];

                    for (uint lane = 0; lane < 4; ++lane) {
                        const uint code = (uint(bits) >> (lane * 2)) & 0x3;
                        const uint xLocal = packed * 4u + lane;
                        const float xr = shared_xRe[xLocal];
                        const float xi = shared_xIm[xLocal];

                        const bool is_real = (code < 2u);
                        const bool raw_pos = bool(code & 1u);
                        const bool imag_pos = CODEBOOK_10_POS_IMAG ? !raw_pos : raw_pos;
                        const bool is_pos = is_real ? raw_pos : imag_pos;

                        const float xrs = is_pos ? xr : -xr;
                        const float xis = is_pos ? xi : -xi;

                        sReXr += is_real ? xrs : 0.0f;
                        sReXi += is_real ? xis : 0.0f;
                        sImXr += is_real ? 0.0f : xrs;
                        sImXi += is_real ? 0.0f : xis;
                    }
                }

                const float gr = float(gammaReal[scaleRowBase + block]);
                const float gi = float(gammaImag[scaleRowBase + block]);
                const float inv_gr = 1.0f / gr;
                const float inv_gi = 1.0f / gi;
                accRe += inv_gr * sReXr + inv_gi * sImXi;
                accIm += inv_gi * sImXr - inv_gr * sReXi;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < OUT_DIM) {
            accRe = simd_sum(accRe);
            accIm = simd_sum(accIm);

            if (lane_id == 0) {
                const uint outIndex = tok * OUT_DIM + row;
                outRe[outIndex] = half(accRe);
                outIm[outIndex] = half(accIm);
            }
        }
    """

    static let widePrefillSource: String = """
        const uint SIMD_W = SIMD_WIDTH;
        const uint PPT = PACKED_PER_THREAD;
        const uint TG = TG_SIZE;

        uint local_id = thread_position_in_grid.x % TG;
        uint group_id = thread_position_in_grid.x / TG;
        uint row_in_group = local_id / SIMD_W;
        uint lane_id = local_id % SIMD_W;
        uint row = group_id * ROWS_PER_GROUP + row_in_group;
        uint tok = thread_position_in_grid.y;

        threadgroup float shared_xRe[QK];
        threadgroup float shared_xIm[QK];

        float accRe = 0.0f;
        float accIm = 0.0f;

        const uint codeRowBase = row * PACKED_PER_ROW;
        const uint scaleRowBase = row * BLOCKS_PER_ROW;
        const uint xTokBase = tok * IN_DIM;

        const device uchar* s_codes[] = {u0Codes, u1Codes, w0Codes, w1Codes};
        const device half* s_gr[] = {u0GammaR, u1GammaR, w0GammaR, w1GammaR};
        const device half* s_gi[] = {u0GammaI, u1GammaI, w0GammaI, w1GammaI};

        for (uint block = 0; block < BLOCKS_PER_ROW; ++block) {
            const uint xBlockBase = xTokBase + block * QK;

            for (uint i = local_id; i < QK; i += TG) {
                shared_xRe[i] = float(xRe[xBlockBase + i]);
                shared_xIm[i] = float(xIm[xBlockBase + i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row < OUT_DIM) {
                const uint codeBlockOff = codeRowBase + block * PACKED_PER_BLOCK;
                const uint scaleOff = scaleRowBase + block;

                for (uint s = 0; s < 4; ++s) {
                    const float im_sign = (s < 2) ? -1.0f : 1.0f;

                    float sReXr = 0.0f, sReXi = 0.0f;
                    float sImXr = 0.0f, sImXi = 0.0f;

                    for (uint p = 0; p < PPT; ++p) {
                        const uint packed = lane_id + p * SIMD_W;
                        const uchar bits = s_codes[s][codeBlockOff + packed];

                        for (uint l = 0; l < 4; ++l) {
                            const uint code = (uint(bits) >> (l * 2)) & 0x3;
                            const uint xLocal = packed * 4u + l;
                            const float xr = shared_xRe[xLocal];
                            const float xi = im_sign * shared_xIm[xLocal];

                            const bool is_real = (code < 2u);
                            const bool raw_pos = bool(code & 1u);
                            const bool imag_pos = CODEBOOK_10_POS_IMAG ? !raw_pos : raw_pos;
                            const bool is_pos = is_real ? raw_pos : imag_pos;

                            const float xrs = is_pos ? xr : -xr;
                            const float xis = is_pos ? xi : -xi;

                            sReXr += is_real ? xrs : 0.0f;
                            sReXi += is_real ? xis : 0.0f;
                            sImXr += is_real ? 0.0f : xrs;
                            sImXi += is_real ? 0.0f : xis;
                        }
                    }

                    const float gr = float(s_gr[s][scaleOff]);
                    const float gi = float(s_gi[s][scaleOff]);
                    const float inv_gr = 1.0f / gr;
                    const float inv_gi = 1.0f / gi;
                    accRe += inv_gr * sReXr + inv_gi * sImXi;
                    accIm += inv_gi * sImXr - inv_gr * sReXi;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < OUT_DIM) {
            accRe = simd_sum(accRe);
            accIm = simd_sum(accIm);

            if (lane_id == 0) {
                const uint outIndex = tok * OUT_DIM + row;
                outRe[outIndex] = half(accRe);
                outIm[outIndex] = half(accIm);
            }
        }
    """
}
