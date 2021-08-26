#include <fstream>
#include <opencv2/opencv.hpp>

#include "ObjectDetector.hpp"

static int num_line = 0;
std::string read_file(const char* path, bool count_line)
{
    std::string result = "";
    std::ifstream f(path);
    //check file exist | true = exist
    if (!f.good()) return result;
    else
    {
        std::string temp;
        while (std::getline(f, result))
        {
            temp.append(result);
            // std::cout << result;
            if (count_line == true)
            {
                num_line++;
            }
        }
        result = temp;
        // Close the file
        f.close();
    }
    return result;
}

int ObjectDetector::init(const char* model_bin, const char* model_param)
{
    
    ncnn::Option opt_detect;
    opt_detect.lightmode = true;
    opt_detect.num_threads = 4;
    opt_detect.blob_allocator = this->g_blob_pool_allocator_detect;
    opt_detect.workspace_allocator = this->g_workspace_pool_allocator_detect;
    // use vulkan compute
    //if (ncnn::get_gpu_count() != 0)
        //opt_detect.use_vulkan_compute = true;
    this->ncnnNet->opt = opt_detect;
    this->ncnnNet->opt.num_threads = 4; //You need to compile with libgomp for multi thread support
    this->ncnnNet->opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support
    this->ncnnNet->opt.use_winograd_convolution = true;
    this->ncnnNet->opt.use_sgemm_convolution = true;
    this->ncnnNet->opt.use_fp16_packed = true;
    this->ncnnNet->opt.use_fp16_storage = true;
    this->ncnnNet->opt.use_fp16_arithmetic = true;
    this->ncnnNet->opt.use_packing_layout = true;
    this->ncnnNet->opt.use_shader_pack8 = false;
    this->ncnnNet->opt.use_image_storage = false;
    int ret = this->ncnnNet->load_param(model_param);
    if (ret != 0)
    {
        printf("Loading model param error\n");
        return ret;        
    }
    
    ret = this->ncnnNet->load_model(model_bin);
    if (ret != 0)
    {
        printf("Loading model bin error\n");
        return ret;
    }
    std::string temp = read_file(model_param, false);
    if (!temp.empty())
    {
        std::regex re("data 0=([0-9]+) 1=([0-9]+) 2=[0-9]{1}");
        std::smatch m;
        std::regex_search(temp, m, re);
        if (!m.empty())
        {
            for (auto i_m : m)
            {
                if (std::regex_match(i_m.str(), re))
                {
                    try
                    {
                        std::string s_w = i_m.str().substr(7, 9);
                        std::string s_h = i_m.str().substr(13, 15);
                        this->width_models = std::stoi(s_w);
                        this->height_models = std::stoi(s_h);
                    }
                    catch(const std::exception& e)
                    {
                        this->width_models = 416;
                        this->height_models = 416;
                    }
                    
                }
            }
        }
    }

    return 0;
}

ObjectDetector::ObjectDetector(std::string& objNameFile, std::string& cfgFile, std::string& weightFile) 
{
    objNames = loadObjectNames(objNameFile);
    
    this->ncnnNet = new ncnn::Net();
    this->g_blob_pool_allocator_detect = new ncnn::UnlockedPoolAllocator();
    this->g_workspace_pool_allocator_detect = new ncnn::PoolAllocator();

    char path_bin[100];
    strcpy(path_bin, weightFile.c_str());
    char path_param[100];
    strcpy(path_param, cfgFile.c_str());
    std::cout << "[INFO] Cfg file: " << path_param << std::endl;
    std::cout << "[INFO] Weight file: " << path_bin << std::endl;
    init(path_bin, path_param);
    std::cout << "[INFO] Init object detector successfully!\n";
}

ObjectDetector::~ObjectDetector() {
    this->ncnnNet->clear();
    this->g_blob_pool_allocator_detect->clear();
    this->g_workspace_pool_allocator_detect->clear();
}

std::vector<std::string> ObjectDetector::loadObjectNames(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::string> fileLines;
    if (!file.is_open()) {
        return fileLines;
    }
    std::string line;
    while (getline(file, line)) {
        line.erase(line.find_last_not_of(" \n\r\t")+1);
        fileLines.push_back(line);
    }
    std::cout << "[INFO] Object names loaded! \n";
    return fileLines;
}

void ObjectDetector::processDetect(const cv::Mat& frame, float thresh, std::vector<bbox_t>& detectResult) 
{
    int img_w = frame.cols;
    int img_h = frame.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows, this->width_models, this->height_models);
    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = this->ncnnNet->create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);
    detectResult.clear();
    for (int i = 0; i < out.h; i++)
    {
       
        const float* values = out.row(i);
 		if(thresh < values[1])
        {
            bbox_t obj;
            obj.obj_id = (int)values[0] - 1;
            obj.prob = values[1];
            obj.x = values[2] * img_w;
            obj.y = values[3] * img_h;
            obj.w = values[4] * img_w - obj.x;
            obj.h = values[5] * img_h - obj.y;
            detectResult.push_back(obj);
            cv::Rect2f rect( obj.x, obj.y, obj.w, obj.h);
        }
        
        //cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 1, 8);
    }
    //cv::imshow("frame", frame);
}
