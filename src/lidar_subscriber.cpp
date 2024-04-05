
// point cloud library install - https://pointclouds.org/downloads/
// https://pointclouds.org/documentation/tutorials/using_pcl_pcl_config.html
// https://www.youtube.com/watch?v=VCobOzw2kHM&t=35s
// https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=9069372&fileOId=9069373

#include <memory>
#include <iostream>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>

class LidarSubscriber : public rclcpp::Node
{
public:
    LidarSubscriber()
        : Node("lidar_subscriber")
    {
        RCLCPP_INFO(this->get_logger(), "node started");

        // set the subscription policy to Best Effort QoS
        auto custom_qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
        custom_qos.best_effort();

        // subscription for raw point cloud data
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points", custom_qos, std::bind(&LidarSubscriber::topic_callback, this, std::placeholders::_1));

        // publisher for the processed point cloud data
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_points", 10);
    }

private:
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {
        auto start = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "Received a point cloud with %u points", msg->width * msg->height);

        // convert ROS2 point cloud message to pcl point cloud object
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // process point cloud data in stages, resulting in points (centroids) that represent cone locations
        auto cropped_cloud = filterPointCloud(cloud);
        auto ground_removed_cloud = removeGroundPlane(cropped_cloud);
        auto clusters = extractClusters(ground_removed_cloud);
        auto centroids = computeCentroids(ground_removed_cloud, clusters);

        // convert and publish the centroids as a ROS2 point cloud message
        sensor_msgs::msg::PointCloud2 centroids_ros_msg;
        pcl::toROSMsg(*centroids, centroids_ros_msg);
        centroids_ros_msg.header.frame_id = "base_footprint";
        centroids_ros_msg.header.stamp = rclcpp::Clock().now();
        publisher_->publish(centroids_ros_msg);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = finish - start;
        RCLCPP_INFO(this->get_logger(), "Callback execution took %f milliseconds", elapsed.count());
    }

    // crops a point cloud to only include points within a defined box
    pcl::PointCloud<pcl::PointXYZ>::Ptr filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) const
    {
        // define bounds of crop box
        double minx = 0.5;
        double miny = -10.0;
        double minz = -5.0;
        double maxx = 20.0;
        double maxy = 10.0;
        double maxz = 2.0;

        // create crop box
        pcl::CropBox<pcl::PointXYZ> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(minx, miny, minz, 1.0));
        boxFilter.setMax(Eigen::Vector4f(maxx, maxy, maxz, 1.0));

        // apply crop box
        boxFilter.setInputCloud(cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        boxFilter.filter(*filtered_cloud);

        return filtered_cloud;
    }

    // uses RANSAC to remove the largest plane (assumed to be the ground) from the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr removeGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) const
    {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);

        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

        if (inliers->indices.empty())
        {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
            // Handle the case where no plane was found
        }
        else
        {
            // Extract the inliers (ground) from the input cloud
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(true); // Set to true to remove the ground plane

            extract.filter(*cloud_filtered);
        }

        return cloud_filtered;
    }

    // uses Euclidian clusters to identify clusters within point cloud
    std::vector<pcl::PointIndices> extractClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) const
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.285); // max size of a cluster
        ec.setMinClusterSize(1);
        ec.setMaxClusterSize(100);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        return cluster_indices;
    }

    // compute the center of each cluster
    pcl::PointCloud<pcl::PointXYZ>::Ptr computeCentroids(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::vector<pcl::PointIndices> &cluster_indices) const
    {
        // Prepare a point cloud to hold the centroids
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroids(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto &cluster : cluster_indices)
        {
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cloud, cluster.indices, centroid);

            // Add the computed centroid to the centroid cloud
            pcl::PointXYZ center(centroid[0], centroid[1], 0.0);
            cloud_centroids->push_back(center);
        }

        return cloud_centroids;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarSubscriber>());
    rclcpp::shutdown();
    return 0;
}
