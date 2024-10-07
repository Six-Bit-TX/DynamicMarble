#! /bin/bash
source ~/.bashrc
conda activate bfe

GPU_ID=0
DATA_ROOT_DIR="/mnt/e/dataset/4drecon/"
DATASETS=(
    # TT
    # sora
    # mars
    GTAV
    )

SCENES=(
    # Family
    # Barn
    # Francis
    # Horse
    # Ignatius
    # santorini
    # LazerTakingOff2
    NightSharkOnHighway
    )

FRAME=(
    # 0000
    # 0001
    # 0002
    # 0003
    # 0004
    # 0005
    # 0006
    # 0007
    # 0008
    # 0009
    # 0010
    # 0011
    # 0012
    # 0013
    # 0014
    # 0015
    # 0016
    # 0017
    # 0018
    # 0019
    # 0020
    # 0021
    # 0022
    # 0023
    # 0024
    # 0025
    # 0026
    # 0027
    # 0028
    # 0029
    # 0030
    # 0031
    # 0032
    # 0033
    # 0034
    # 0035
    # 0036
    # 0037
    # 0038
    # 0039
    # 0040
    # 0041
    # 0042
    # 0043
    # 0044
    # 0045
    # 0046
    # 0047
    # 0048
    # 0049
    # 0050
    # 0051
    # 0052
    # 0053
    # 0054
    # 0055
    # 0056
    # 0057
    # 0058
    # 0059
    # 0060
    # 0061
    # 0062
    # 0063
    # 0064
    # 0065
    # 0066
    # 0067
    # 0068
    # 0069
    # 0070
    # 0071
    # 0072
    # 0073
    # 0074
    # 0075
    # 0076
    # 0077
    # 0078
    # 0079
    # 0080
    # 0081
    # 0082
    # 0083
    # 0084
    # 0085
    # 0086
    # 0087
    # 0088
    # 0089
    # 0090
    # 0091
    # 0092
    # 0093
    # 0094
    # 0095
    # 0096
    # 0097
    # 0098
    # 0099
    # 0100
    # 0101
    # 0102
    # 0103
    # 0104
    # 0105
    # 0106
    # 0107
    # 0108
    # 0109
    # 0110
    # 0111
    # 0112
    # 0113
    # 0114
    # 0115
    # 0116
    # 0117
    # 0118
    # 0119
    # 0120
    # 0121
    # 0122
    # 0123
    # 0124
    # 0125
    # 0126
    # 0127
    # 0128
    # 0129
    # 0130
    # 0131
    # 0132
    # 0133
    # 0134
    # 0135
    # 0136
    # 0137
    # 0138
    # 0139
    # 0140
    # 0141
    # 0142
    # 0143
    # 0144
    # 0145
    # 0146
    # 0147
    # 0148
    # 0149
    # 0150
    # 0151
    # 0152
    # 0153
    # 0154
    # 0155
    # 0156
    # 0157
    # 0158
    # 0159
    # 0160
    # 0161
    # 0162
    # 0163
    # 0164
    # 0165
    # 0166
    # 0167
    # 0168
    # 0169
    # 0170
    # 0171
    # 0172
    # 0173
    # 0174
    # 0175
    # 0176
    # 0177
    # 0178
    # 0179
    # 0180
    # 0181
    # 0182
    # 0183
    # 0184
    # 0185
    # 0186
    # 0187
    # 0188
    # 0189
    # 0190
    # 0191
    # 0192
    # 0193
    # 0194
    # 0195
    # 0196
    # 0197
    # 0198
    # 0199
    0200
    # 0201
    # 0202
    # 0203
    # 0204
    # 0205
    # 0206
    # 0207
    # 0208
    # 0209
    # 0210
    # 0211
    # 0212
    # 0213
    # 0214
    # 0215
    # 0216
    # 0217
    # 0218
    # 0219
    # 0220
    # 0221
    # 0222
    # 0223
    # 0224
    # 0225
    # 0226
    # 0227
    # 0228
    # 0229
    # 0230
    # 0231
    # 0232
    # 0233
    # 0234
    # 0235
    # 0236
    # 0237
    # 0238
    # 0239
    # 0240
    # 0241
    # 0242
    # 0243
    # 0244
    # 0245
    # 0246
    # 0247
    # 0248
    # 0249
    0250
    # 0251
    # 0252
    # 0253
    # 0254
    # 0255
    # 0256
    # 0257
    # 0258
    # 0259
    # 0260
    # 0261
    # 0262
    # 0263
    # 0264
    # 0265
    # 0266
    # 0267
    # 0268
    # 0269
    # 0270
    # 0271
    # 0272
    # 0273
    # 0274
    # 0275
    # 0276
    # 0277
    # 0278
    # 0279
    # 0280
    # 0281
    # 0282
    # 0283
    # 0284
    # 0285
    # 0286
    # 0287
    # 0288
    # 0289
    # 0290
    # 0291
    # 0292
    # 0293
    # 0294
    # 0295
    # 0296
    # 0297
    # 0298
    0299
    )

N_VIEW=5

# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=3000
pose_lr=1x

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_FRAME in "${FRAME[@]}"; do

            # SOURCE_PATH must be Absolute path
            SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/frame_${N_FRAME}
            MODEL_PATH=./output/infer/${DATASET}/${SCENE}/frame_${N_FRAME}_${gs_train_iter}Iter_${pose_lr}PoseLR_marble/

            # # ----- (1) Dust3r_coarse_geometric_initialization -----
            CMD_D1="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./coarse_init_infer.py \
            --img_base_path ${SOURCE_PATH} \
            --n_views ${N_VIEW}  \
            --focal_avg \
            "

            # # ----- (2) Train: jointly optimize pose -----
            CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./train_joint.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            --n_views ${N_VIEW}  \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --optim_pose \
            "

            # ----- (3) Render interpolated pose & output video -----
            CMD_RI="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./render_by_interp.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            --n_views ${N_VIEW}  \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --eval \
            --get_video \
            "


            # echo "========= ${SCENE}: Dust3r_coarse_geometric_initialization ========="
            # eval $CMD_D1
            # echo "========= ${SCENE}: Train: jointly optimize pose ========="
            # eval $CMD_T
            echo "========= ${SCENE}: Render interpolated pose & output video ========="
            eval $CMD_RI
            done
        done
    done