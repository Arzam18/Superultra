#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <cstring>
#include "incbin/incbin.h"
#include "nnue.h"
#include "types.h"
#include "helpers.h"

INCBIN(nnueNet, "nnueweights.bin");

alignas(32) static NNUEWeight W1[INPUT_HALF * HIDDEN_HALF];
alignas(32) static NNUEWeight B1[HIDDEN_HALF];
alignas(32) static NNUEWeight W2[OUTPUT_WEIGHT_BUCKET_COUNT * HIDDEN_HALF * 2];
alignas(32) static NNUEWeight B2[OUTPUT_WEIGHT_BUCKET_COUNT];

void NeuralNetwork::addFeature(Piece pieceType, Square sq, Color col, Square wking, Square bking){
    const int whitePerspIdx = getInputIndex(pieceType, col, sq, WHITE, wking) * HIDDEN_HALF;
    const int blackPerspIdx = getInputIndex(pieceType, col, sq, BLACK, bking) * HIDDEN_HALF;

#if defined(__aarch64__) || defined(__ARM_NEON)
    const auto vectorAccumWhitePtr = reinterpret_cast<int16_t*>(&accum[0]);
    const auto vectorWeightWhitePtr = reinterpret_cast<const int16_t*>(&W1[whitePerspIdx]);
    const auto vectorAccumBlackPtr = reinterpret_cast<int16_t*>(&accum[HIDDEN_HALF]);
    const auto vectorWeightBlackPtr = reinterpret_cast<const int16_t*>(&W1[blackPerspIdx]);

    for (int i = 0; i < HIDDEN_HALF; i += 8){
        vst1q_s16(vectorAccumWhitePtr + i,
                  vaddq_s16(vld1q_s16(vectorAccumWhitePtr + i),
                            vld1q_s16(vectorWeightWhitePtr + i)));
        vst1q_s16(vectorAccumBlackPtr + i,
                  vaddq_s16(vld1q_s16(vectorAccumBlackPtr + i),
                            vld1q_s16(vectorWeightBlackPtr + i)));
    }
#elif defined(__AVX__) || defined(__AVX2__)
    const auto vectorAccumWhitePtr = reinterpret_cast<__m256i*>(&accum[0]);
    const auto vectorWeightWhitePtr = reinterpret_cast<__m256i*>(&W1[whitePerspIdx]);
    const auto vectorAccumBlackPtr = reinterpret_cast<__m256i*>(&accum[HIDDEN_HALF]);
    const auto vectorWeightBlackPtr = reinterpret_cast<__m256i*>(&W1[blackPerspIdx]);

    for (int i = 0; i < HIDDEN_HALF / 16; i++){
        vectorAccumWhitePtr[i] = _mm256_add_epi16(vectorAccumWhitePtr[i], vectorWeightWhitePtr[i]);
        vectorAccumBlackPtr[i] = _mm256_add_epi16(vectorAccumBlackPtr[i], vectorWeightBlackPtr[i]);
    }
#else
    for (int i = 0; i < HIDDEN_HALF; i++){
        accum[i] += W1[whitePerspIdx + i];
        accum[i + HIDDEN_HALF] += W1[blackPerspIdx + i];
    }
#endif
}

void NeuralNetwork::removeFeature(Piece pieceType, Square sq, Color col, Square wking, Square bking){
    const int whitePerspIdx = getInputIndex(pieceType, col, sq, WHITE, wking) * HIDDEN_HALF;
    const int blackPerspIdx = getInputIndex(pieceType, col, sq, BLACK, bking) * HIDDEN_HALF;

#if defined(__aarch64__) || defined(__ARM_NEON)
    const auto vectorAccumWhitePtr = reinterpret_cast<int16_t*>(&accum[0]);
    const auto vectorWeightWhitePtr = reinterpret_cast<const int16_t*>(&W1[whitePerspIdx]);
    const auto vectorAccumBlackPtr = reinterpret_cast<int16_t*>(&accum[HIDDEN_HALF]);
    const auto vectorWeightBlackPtr = reinterpret_cast<const int16_t*>(&W1[blackPerspIdx]);

    for (int i = 0; i < HIDDEN_HALF; i += 8){
        vst1q_s16(vectorAccumWhitePtr + i,
                  vsubq_s16(vld1q_s16(vectorAccumWhitePtr + i),
                            vld1q_s16(vectorWeightWhitePtr + i)));
        vst1q_s16(vectorAccumBlackPtr + i,
                  vsubq_s16(vld1q_s16(vectorAccumBlackPtr + i),
                            vld1q_s16(vectorWeightBlackPtr + i)));
    }
#elif defined(__AVX__) || defined(__AVX2__)
    const auto vectorAccumWhitePtr = reinterpret_cast<__m256i*>(&accum[0]);
    const auto vectorWeightWhitePtr = reinterpret_cast<__m256i*>(&W1[whitePerspIdx]);
    const auto vectorAccumBlackPtr = reinterpret_cast<__m256i*>(&accum[HIDDEN_HALF]);
    const auto vectorWeightBlackPtr = reinterpret_cast<__m256i*>(&W1[blackPerspIdx]);

    for (int i = 0; i < HIDDEN_HALF / 16; i++){
        vectorAccumWhitePtr[i] = _mm256_sub_epi16(vectorAccumWhitePtr[i], vectorWeightWhitePtr[i]);
        vectorAccumBlackPtr[i] = _mm256_sub_epi16(vectorAccumBlackPtr[i], vectorWeightBlackPtr[i]);
    }
#else
    for (int i = 0; i < HIDDEN_HALF; i++){
        accum[i] -= W1[whitePerspIdx + i];
        accum[i + HIDDEN_HALF] -= W1[blackPerspIdx + i];
    }
#endif
}

void NeuralNetwork::updateMove(Piece pieceType, Square st, Square en, Color col, Square wking, Square bking){
    const int whitePerspStIdx = getInputIndex(pieceType, col, st, WHITE, wking) * HIDDEN_HALF;
    const int blackPerspEnIdx = getInputIndex(pieceType, col, en, BLACK, bking) * HIDDEN_HALF;
    const int whitePerspEnIdx = getInputIndex(pieceType, col, en, WHITE, wking) * HIDDEN_HALF;
    const int blackPerspStIdx = getInputIndex(pieceType, col, st, BLACK, bking) * HIDDEN_HALF;

#if defined(__aarch64__) || defined(__ARM_NEON)
    const auto vectorAccumWhitePtr = reinterpret_cast<int16_t*>(&accum[0]);
    const auto vectorWeightStWhitePtr = reinterpret_cast<const int16_t*>(&W1[whitePerspStIdx]);
    const auto vectorWeightEnWhitePtr = reinterpret_cast<const int16_t*>(&W1[whitePerspEnIdx]);
    const auto vectorAccumBlackPtr = reinterpret_cast<int16_t*>(&accum[HIDDEN_HALF]);
    const auto vectorWeightStBlackPtr = reinterpret_cast<const int16_t*>(&W1[blackPerspStIdx]);
    const auto vectorWeightEnBlackPtr = reinterpret_cast<const int16_t*>(&W1[blackPerspEnIdx]);

    for (int i = 0; i < HIDDEN_HALF; i += 8){
        const int16x8_t whiteDelta = vsubq_s16(vld1q_s16(vectorWeightEnWhitePtr + i),
                                                vld1q_s16(vectorWeightStWhitePtr + i));
        const int16x8_t blackDelta = vsubq_s16(vld1q_s16(vectorWeightEnBlackPtr + i),
                                                vld1q_s16(vectorWeightStBlackPtr + i));
        vst1q_s16(vectorAccumWhitePtr + i,
                  vaddq_s16(vld1q_s16(vectorAccumWhitePtr + i), whiteDelta));
        vst1q_s16(vectorAccumBlackPtr + i,
                  vaddq_s16(vld1q_s16(vectorAccumBlackPtr + i), blackDelta));
    }
#elif defined(__AVX__) || defined(__AVX2__)
    const auto vectorAccumWhitePtr = reinterpret_cast<__m256i*>(&accum[0]);
    const auto vectorWeightStWhitePtr = reinterpret_cast<__m256i*>(&W1[whitePerspStIdx]);
    const auto vectorWeightEnWhitePtr = reinterpret_cast<__m256i*>(&W1[whitePerspEnIdx]);
    const auto vectorAccumBlackPtr = reinterpret_cast<__m256i*>(&accum[HIDDEN_HALF]);
    const auto vectorWeightStBlackPtr = reinterpret_cast<__m256i*>(&W1[blackPerspStIdx]);
    const auto vectorWeightEnBlackPtr = reinterpret_cast<__m256i*>(&W1[blackPerspEnIdx]);

    for (int i = 0; i < HIDDEN_HALF / 16; i++){
        vectorAccumWhitePtr[i] = _mm256_add_epi16(
            vectorAccumWhitePtr[i],
            _mm256_sub_epi16(vectorWeightEnWhitePtr[i], vectorWeightStWhitePtr[i]));
        vectorAccumBlackPtr[i] = _mm256_add_epi16(
            vectorAccumBlackPtr[i],
            _mm256_sub_epi16(vectorWeightEnBlackPtr[i], vectorWeightStBlackPtr[i]));
    }
#else
    for (int i = 0; i < HIDDEN_HALF; i++){
        accum[i] += -W1[whitePerspStIdx + i] + W1[whitePerspEnIdx + i];
        accum[i + HIDDEN_HALF] += -W1[blackPerspStIdx + i] + W1[blackPerspEnIdx + i];
    }
#endif
}

void NeuralNetwork::refresh(Piece *board, Square wking, Square bking){
#if defined(__aarch64__) || defined(__ARM_NEON)
    const auto vectorBiasPtr = reinterpret_cast<const int16_t*>(&B1[0]);
    auto vectorAccumTopPtr = reinterpret_cast<int16_t*>(&accum[0]);
    auto vectorAccumBotPtr = reinterpret_cast<int16_t*>(&accum[HIDDEN_HALF]);

    for (int i = 0; i < HIDDEN_HALF; i += 8){
        const int16x8_t bias = vld1q_s16(vectorBiasPtr + i);
        vst1q_s16(vectorAccumTopPtr + i, bias);
        vst1q_s16(vectorAccumBotPtr + i, bias);
    }
#elif defined(__AVX__) || defined(__AVX2__)
    const auto vectorAccumTopPtr = reinterpret_cast<__m256i*>(&accum[0]);
    const auto vectorAccumBotPtr = reinterpret_cast<__m256i*>(&accum[HIDDEN_HALF]);
    const auto vectorBiasPtr = reinterpret_cast<__m256i*>(&B1[0]);

    for (int i = 0; i < HIDDEN_HALF / 16; i++){
        vectorAccumTopPtr[i] = vectorBiasPtr[i];
        vectorAccumBotPtr[i] = vectorBiasPtr[i];
    }
#else
    for (int i = 0; i < HIDDEN_HALF; i++){
        accum[i] = B1[i];
        accum[i + HIDDEN_HALF] = B1[i];
    }
#endif

    for (Square sq = 0; sq < 64; sq++){
        if (board[sq] != NO_PIECE){
            addFeature(getPieceType(board[sq]), sq, getPieceColor(board[sq]), wking, bking);
        }
    }
}

Score NeuralNetwork::eval(Color col, int8 pieceCount){
    const int topIdx = (col == WHITE ? 0 : HIDDEN_HALF);
    const int botIdx = (col == WHITE ? HIDDEN_HALF : 0);
    const int outputWeightBucket = calculateOutputBucket(pieceCount);
    const int outputWeightsIndex = outputWeightBucket * HIDDEN_HALF * 2;
    int eval = B2[outputWeightBucket];

#if defined(__aarch64__) || defined(__ARM_NEON)
    int32x4_t vectorEval = vdupq_n_s32(0);
    const int16x8_t vectorCreluL = vdupq_n_s16(CRELU_L);
    const int16x8_t vectorCreluR = vdupq_n_s16(CRELU_R);
    const auto vectorAccumTopPtr = reinterpret_cast<const int16_t*>(&accum[topIdx]);
    const auto vectorWeightTopPtr = reinterpret_cast<const int16_t*>(&W2[outputWeightsIndex]);
    const auto vectorAccumBotPtr = reinterpret_cast<const int16_t*>(&accum[botIdx]);
    const auto vectorWeightBotPtr = reinterpret_cast<const int16_t*>(&W2[outputWeightsIndex + HIDDEN_HALF]);

    for (int i = 0; i < HIDDEN_HALF; i += 8){
        const int16x8_t top = vminq_s16(vmaxq_s16(vld1q_s16(vectorAccumTopPtr + i), vectorCreluL), vectorCreluR);
        const int16x8_t topW = vld1q_s16(vectorWeightTopPtr + i);
        const int16x8_t bot = vminq_s16(vmaxq_s16(vld1q_s16(vectorAccumBotPtr + i), vectorCreluL), vectorCreluR);
        const int16x8_t botW = vld1q_s16(vectorWeightBotPtr + i);

        const int32x4_t topLo = vmull_s16(vget_low_s16(top), vget_low_s16(topW));
        const int32x4_t topHi = vmull_s16(vget_high_s16(top), vget_high_s16(topW));
        const int32x4_t botLo = vmull_s16(vget_low_s16(bot), vget_low_s16(botW));
        const int32x4_t botHi = vmull_s16(vget_high_s16(bot), vget_high_s16(botW));

        vectorEval = vaddq_s32(vectorEval, topLo);
        vectorEval = vaddq_s32(vectorEval, topHi);
        vectorEval = vaddq_s32(vectorEval, botLo);
        vectorEval = vaddq_s32(vectorEval, botHi);
    }

    eval += vget_lane_s32(vpadd_s32(vget_low_s32(vectorEval), vget_high_s32(vectorEval)), 0);
    eval += vget_lane_s32(vpadd_s32(vget_low_s32(vectorEval), vget_high_s32(vectorEval)), 1);
#elif defined(__AVX__) || defined(__AVX2__)
    auto vectorEval = _mm256_setzero_si256();
    const auto vectorCreluL = _mm256_set1_epi16(CRELU_L);
    const auto vectorCreluR = _mm256_set1_epi16(CRELU_R);
    const auto vectorAccumTopPtr = reinterpret_cast<__m256i*>(&accum[topIdx]);
    const auto vectorWeightTopPtr = reinterpret_cast<__m256i*>(&W2[outputWeightsIndex]);
    const auto vectorAccumBotPtr = reinterpret_cast<__m256i*>(&accum[botIdx]);
    const auto vectorWeightBotPtr = reinterpret_cast<__m256i*>(&W2[outputWeightsIndex + HIDDEN_HALF]);

    for (int i = 0; i < HIDDEN_HALF / 16; i++){
        vectorEval = _mm256_add_epi32(vectorEval,
            _mm256_madd_epi16(_mm256_min_epi16(_mm256_max_epi16(vectorAccumTopPtr[i], vectorCreluL), vectorCreluR), vectorWeightTopPtr[i]));
    }
    for (int i = 0; i < HIDDEN_HALF / 16; i++){
        vectorEval = _mm256_add_epi32(vectorEval,
            _mm256_madd_epi16(_mm256_min_epi16(_mm256_max_epi16(vectorAccumBotPtr[i], vectorCreluL), vectorCreluR), vectorWeightBotPtr[i]));
    }
    for (int i = 0; i < 8; i++){
        eval += _mm256_extract_epi32(vectorEval, i);
    }
#else
    for (int i = 0; i < HIDDEN_HALF; i++){
        int16 input = accum[topIdx + i];
        int16 weight = W2[outputWeightsIndex + i];
        eval += std::clamp(input, CRELU_L, CRELU_R) * weight;
    }
    for (int i = 0; i < HIDDEN_HALF; i++){
        int16 input = accum[botIdx + i];
        int16 weight = W2[outputWeightsIndex + HIDDEN_HALF + i];
        eval += std::clamp(input, CRELU_L, CRELU_R) * weight;
    }
#endif

    eval = ((eval * evalScale) / (Q1 * Q2));
    return static_cast<Score>(eval);
}

void initNNUEWeights(){
    int idx = 0;

    // W1
    memcpy(W1, gnnueNetData + idx, INPUT_HALF * HIDDEN_HALF * sizeof(NNUEWeight));
    idx += INPUT_HALF * HIDDEN_HALF * sizeof(NNUEWeight);

    // B1
    memcpy(B1, gnnueNetData + idx, HIDDEN_HALF * sizeof(NNUEWeight));
    idx += HIDDEN_HALF * sizeof(NNUEWeight);

    // W2
    memcpy(W2, gnnueNetData + idx, OUTPUT_WEIGHT_BUCKET_COUNT * HIDDEN_HALF * 2 * sizeof(NNUEWeight));
    idx += HIDDEN_HALF * sizeof(NNUEWeight) * 2;
    
    // B2
    memcpy(B2, gnnueNetData + idx, OUTPUT_WEIGHT_BUCKET_COUNT * sizeof(NNUEWeight));
}