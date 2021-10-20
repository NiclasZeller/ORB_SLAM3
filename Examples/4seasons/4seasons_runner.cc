#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>
#include "ImuTypes.h"

using namespace std;

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

double ttrack_tot = 0;
int main(int argc, char **argv) {

    if(argc < 4) {
        cerr << endl << "Usage: ./stereo_inertial_tum_vi path_to_vocabulary path_to_settings path_to_dataset_folder\n";
        return 1;
    }

    bool useImu = true;
    bool useGui = true;
    bool useRectImages = true;

    std::string dirDataset(argv[3]);

    std::string dirCamLeft, dirCamRight;
    if(useRectImages) {
        dirCamLeft = dirDataset + "/undistorted_images/cam0";
        dirCamRight = dirDataset + "/undistorted_images/cam1";
    }
    else {
        dirCamLeft = dirDataset + "/distorted_images/cam0";
        dirCamRight = dirDataset + "/distorted_images/cam1";
    }


    std::string fileTimestamps = dirDataset + "/times.txt";
    std::string fileImu = dirDataset + "/imu.txt";

    std::vector<std::string> vstrImageLeftFilenames, vstrImageRightFilenames;
    std::vector<double> vTimestampsCam;
    std::vector<cv::Point3f> vAcc, vGyro;
    std::vector<double> vTimestampsImu;
    int nImages, nImu, firstImuIdx;
    nImages = nImu = firstImuIdx = 0;

    std::cout << "Loading images ...\n";
    LoadImages(dirCamLeft, dirCamRight, fileTimestamps, vstrImageLeftFilenames, vstrImageRightFilenames, vTimestampsCam);
    std::cout << "Total images: " << vstrImageLeftFilenames.size() << "\n";
    std::cout << "Total cam ts: " << vTimestampsCam.size() << "\n";
    std::cout << "first cam ts: " << vTimestampsCam[0] << "\n";
    std::cout << "LOADED!\n";
    nImages = vstrImageLeftFilenames.size();

    if(nImages<=0) {
        cerr << "ERROR: Failed to load images.\n";
        return 1;
    }

    if(useImu) {
        std::cout << "Loading IMU data ...\n";
        LoadIMU(fileImu, vTimestampsImu, vAcc, vGyro);
        std::cout << "Total IMU meas: " << vTimestampsImu.size() << "\n";
        std::cout << "first IMU ts: " << vTimestampsImu[0] << "\n";
        std::cout << "LOADED!\n";
        nImu = vTimestampsImu.size();

        if (nImu<=0) {
            cerr << "ERROR: Failed to load IMU measurements.\n";
            return 1;
        }
        while(vTimestampsImu[firstImuIdx]<vTimestampsCam[0]) {
            firstImuIdx++;
        }
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << "\n-------\n";
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System::eSensor sensorType;
    if(useImu) {
        sensorType = ORB_SLAM3::System::IMU_STEREO;
    }
    else {
        sensorType = ORB_SLAM3::System::STEREO;
    }
    ORB_SLAM3::System SLAM(argv[1],argv[2],sensorType, useGui, 0, std::string(), std::string(), true);

    cv::Mat imLeft, imRight;
    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    int proccIm = 0;

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    for(int ni=0; ni<nImages; ni++, proccIm++) {

        // Read image from file
        imLeft = cv::imread(vstrImageLeftFilenames[ni],cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRightFilenames[ni],cv::IMREAD_UNCHANGED);

        // clahe
        clahe->apply(imLeft,imLeft);
        clahe->apply(imRight,imRight);

        double tframe = vTimestampsCam[ni];

        if(imLeft.empty() or imRight.empty()) {
            cerr << endl << "Failed to load image at: " <<  vstrImageLeftFilenames[ni] << "\n";
            return 1;
        }

        // Load imu measurements from previous frame
        vImuMeas.clear();
        if(useImu) {
            if (ni > 0) {
                while (vTimestampsImu[firstImuIdx] <= vTimestampsCam[ni]) {
                    vImuMeas.push_back(
                        ORB_SLAM3::IMU::Point(vAcc[firstImuIdx].x, vAcc[firstImuIdx].y, vAcc[firstImuIdx].z,
                                              vGyro[firstImuIdx].x, vGyro[firstImuIdx].y, vGyro[firstImuIdx].z,
                                              vTimestampsImu[firstImuIdx]));
                    firstImuIdx++;
                }
            }
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        if(useImu) {
            SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);
        }
        else {
            SLAM.TrackStereo(imLeft,imRight,tframe);
        }


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        ttrack_tot += ttrack;

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1) {
            T = vTimestampsCam[ni+1]-tframe;
        }
        else if(ni>0) {
            T = tframe-vTimestampsCam[ni-1];
        }

        if(ttrack<T) {
            usleep((T-ttrack)*1e6);
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    std::chrono::system_clock::time_point scNow = std::chrono::system_clock::now();
    std::time_t now = std::chrono::system_clock::to_time_t(scNow);
    std::stringstream ss;
    ss << now;

//    if (bFileName)
//    {
//        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
//        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
//        SLAM.SaveTrajectoryEuRoC(f_file);
//        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
//    }
//    else
//    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
//    }

    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/proccIm << endl;

    return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    cout << strPathLeft << endl;
    cout << strPathRight << endl;
    cout << strPathTimes << endl;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(30000);
    vstrImageLeft.reserve(30000);
    vstrImageRight.reserve(30000);

    int64_t t_ns;
    double t_sec, exposure_ms;

    std::string line;
    while (std::getline(fTimes, line)) {
        if (line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> t_ns >> t_sec >> exposure_ms;

        std::string imageFile = std::to_string(t_ns) + ".png";
        std::cout << "load image " << imageFile << "\n";

        vstrImageLeft.push_back(strPathLeft + "/" + imageFile);
        vstrImageRight.push_back(strPathRight + "/" + imageFile);
        // should be more precise than the timestamp stored in seconds in the file
        vTimeStamps.push_back(static_cast<double>(t_ns)/1.0e9);
    }
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(100000);
    vAcc.reserve(100000);
    vGyro.reserve(100000);

    std::string line;
    while (std::getline(fImu, line)) {
        if (line[0] == '#') continue;

        std::stringstream ss(line);

        uint64_t t_ns;
        cv::Point3f acc, gyro;
        ss >> t_ns >> gyro.x >> gyro.y >> gyro.z >> acc.x >> acc.y >> acc.z;

        vTimeStamps.push_back(static_cast<double>(t_ns)/1.0e9);
        vAcc.push_back(acc);
        vGyro.push_back(gyro);
    }
}
