// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLOX_H
#define YOLOX_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <array>

#include <net.h>

#include "line2Dup.h"


class Yolox
{
public:
    Yolox();

    int load(AAssetManager* mgr, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int draw(cv::Mat& rgb);
    int predict(cv::Mat& rgb);
    int multi_predict(cv::Mat img);

private:
    ncnn::Net classifer;

    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    int image_w;
    int image_h;
    int in_w;
    int in_h;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;

    static constexpr size_t num_classes = 10;
    // 模板图片
    cv::Mat templ_img;
    int num_feature = 150;
    line2Dup::Detector detector;
};

#endif // NANODET_H
