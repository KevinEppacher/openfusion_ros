#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class CameraPosePublisher : public rclcpp::Node
{
public:
    explicit CameraPosePublisher(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
    void publishPose();

    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr posePublisher;
    rclcpp::TimerBase::SharedPtr timer;

    std::string parentFrame;
    std::string childFrame;
    double rateHz;
};
