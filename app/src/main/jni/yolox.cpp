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

#include "yolox.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cpu.h"

using namespace cv;


// ****************************************** borrow from shape_based_matching begin ************************************************ //
namespace  cv_dnn {
    namespace
    {

        template <typename T>
        static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                                const std::pair<float, T>& pair2)
        {
            return pair1.first > pair2.first;
        }

    } // namespace

    inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                                 std::vector<std::pair<float, int> >& score_index_vec)
    {
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] > threshold)
            {
                score_index_vec.push_back(std::make_pair(scores[i], i));
            }
        }
        std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                         SortScorePairDescend<int>);
        if (top_k > 0 && top_k < (int)score_index_vec.size())
        {
            score_index_vec.resize(top_k);
        }
    }

    template <typename BoxType>
    inline void NMSFast_(const std::vector<BoxType>& bboxes,
                         const std::vector<float>& scores, const float score_threshold,
                         const float nms_threshold, const float eta, const int top_k,
                         std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
    {
        CV_Assert(bboxes.size() == scores.size());
        std::vector<std::pair<float, int> > score_index_vec;
        GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

        // Do nms.
        float adaptive_threshold = nms_threshold;
        indices.clear();
        for (size_t i = 0; i < score_index_vec.size(); ++i) {
            const int idx = score_index_vec[i].second;
            bool keep = true;
            for (int k = 0; k < (int)indices.size() && keep; ++k) {
                const int kept_idx = indices[k];
                float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            }
            if (keep)
                indices.push_back(idx);
            if (keep && eta < 1 && adaptive_threshold > 0.5) {
                adaptive_threshold *= eta;
            }
        }
    }


// copied from opencv 3.4, not exist in 3.0
    template<typename _Tp> static inline
    double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
        _Tp Aa = a.area();
        _Tp Ab = b.area();

        if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
            // jaccard_index = 1 -> distance = 0
            return 0.0;
        }

        double Aab = (a & b).area();
        // distance = 1 - jaccard_index
        return 1.0 - Aab / (Aa + Ab - Aab);
    }

    template <typename T>
    static inline float rectOverlap(const T& a, const T& b)
    {
        return 1.f - static_cast<float>(jaccardDistance__(a, b));
    }

    void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                  const float score_threshold, const float nms_threshold,
                  std::vector<int>& indices, const float eta=1, const int top_k=0)
    {
        NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
    }

}
// ****************************************** borrow from shape_based_matching end ************************************************ //

Yolox::Yolox()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolox::load(AAssetManager* mgr, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    classifer.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();


    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    classifer.opt = ncnn::Option();
#if NCNN_VULKAN
    classifer.opt.use_vulkan_compute = use_gpu;
#endif
    classifer.opt.num_threads = ncnn::get_big_cpu_count();
    classifer.opt.blob_allocator = &blob_pool_allocator;
    classifer.opt.workspace_allocator = &workspace_pool_allocator;

    // 加载mobilenetv2模型
    classifer.load_param(mgr, "mobilenetv2.param");
    classifer.load_model(mgr, "mobilenetv2.bin");

    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    // 读取模板图片
    AAsset* asset = AAssetManager_open(mgr, "circle_templ.png", AASSET_MODE_BUFFER);
    const void* data = AAsset_getBuffer(asset);
    int length = AAsset_getLength(asset);

    std::vector<char> buffer(length);
    memcpy(buffer.data(), data, length);
    templ_img = imdecode(buffer, IMREAD_UNCHANGED);

    AAsset_close(asset);

    // 初始化模板匹配检测器
    shape_based_matching::shapeInfo_producer shapes(templ_img);

    shapes.scale_range = {0.1f, 1};
    shapes.scale_step = 0.01f;
    shapes.produce_infos();

    std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
    std::string class_id = "circle";
    for(auto& info: shapes.infos){

        // template img, id, mask,
        //feature numbers(missing it means using the detector initial num)
        int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info),
                                            int(num_feature*info.scale));

        // may fail when asking for too many feature_nums for small training img
        if(templ_id != -1){  // only record info when we successfully add template
            infos_have_templ.push_back(info);
        }
    }

    return 0;
}

int Yolox::multi_predict(cv::Mat img)
{
    size_t bins[num_classes] = {0};
    for(size_t flip_code = 0; flip_code < 3; flip_code++) {
        cv::Mat img_flip = img.clone();
        cv::flip(img, img_flip, flip_code);
        int pred_cls = predict(img_flip);
        bins[pred_cls-1]++;
    }

    int min_cnt = 0;
    int voted_pred_cls = -1;
    for(size_t i = 0; i < num_classes; i++){
        if(bins[i] > min_cnt){
            min_cnt = bins[i];
            voted_pred_cls = i;
        }
    }
    return voted_pred_cls + 1;
}

int Yolox::predict(cv::Mat &img)
{
    int img_w = img.cols;
    int img_h = img.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, 224, 224);
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = classifer.create_extractor();
    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);
    const float* feat_ptr = out.channel(0);

    float tmp_prob = FLT_MIN;
    int pred_cls = -1;

    for(size_t i =0; i < num_classes; i++){
        if(feat_ptr[i] > tmp_prob) {
            pred_cls = i;
            tmp_prob = feat_ptr[i];
        }
    }

    return pred_cls + 1;
}

int Yolox::draw(cv::Mat& rgb)
{
    // 模板匹配算法检测
    std::vector<std::string> ids;

    // make the img having 32*n width & height
    // at least 16*n here for two pyrimads with strides 4 8
    int stride = 32;
    int n = rgb.rows/stride;
    int m = rgb.cols/stride;
    Rect roi(0, 0, stride*m , stride*n);
    Mat img = rgb(roi).clone();

    auto matches = detector.match(img, 95, ids);

    // NMS后处理
    std::vector<Rect> boxes;
    std::vector<float> scores;
    std::vector<int> idxs;
    for(auto match: matches){
        Rect box;
        box.x = match.x;
        box.y = match.y;

        auto templ = detector.getTemplates("circle",
                                           match.template_id);

        box.width = templ[0].width;
        box.height = templ[0].height;
        boxes.push_back(box);
        scores.push_back(match.similarity);
    }
    cv_dnn::NMSBoxes(boxes, scores, 0, 0.2f, idxs);

    int total_score = 0;

    for(auto i: idxs){
        auto match = matches[i];
        auto templ = detector.getTemplates("circle",
                                           match.template_id);
        // template:
        // nums: num_pyramids * num_modality (modality, depth or RGB, always 1 here)
        // template[0]: lowest pyrimad(more pixels)
        // template[0].width: actual width of the matched template
        // template[0].tl_x / tl_y: topleft corner when cropping templ during training
        // In this case, we can regard width/2 = radius

        cv::Mat crop = rgb(cv::Rect(match.x, match.y, templ[0].width, templ[0].height));
        cv::Mat crop_bgr;
        cv::cvtColor(crop, crop_bgr, cv::COLOR_BGR2RGB);

        // int pred_cls = predict(crop_bgr);  // TODO
        int pred_cls = multi_predict(crop_bgr);
        total_score += pred_cls;

        cv::putText(rgb, std::to_string(pred_cls),
                        Point(match.x + templ[0].width - 10, match.y - 3), FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0,255,0));
    }

    char text[32];
    sprintf(text, " total_score=%3d ", total_score);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0));

    return 0;
}