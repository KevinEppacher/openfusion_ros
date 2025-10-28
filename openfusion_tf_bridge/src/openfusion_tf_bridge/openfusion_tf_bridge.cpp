#include "openfusion_tf_bridge/openfusion_tf_bridge_node.hpp"

CameraPosePublisher::CameraPosePublisher(const rclcpp::NodeOptions &options)
    : Node("camera_pose_publisher", options)
{
    // Declare parameters
    declare_parameter("robot.map_frame", "map");
    declare_parameter("robot.camera_frame", "camera_link");
    declare_parameter("rate_hz", 30.0);
    declare_parameter("camera_pose_topic", "camera_pose");

    parentFrame = get_parameter("robot.map_frame").as_string();
    childFrame  = get_parameter("robot.camera_frame").as_string();
    rateHz      = get_parameter("rate_hz").as_double();
    std::string poseTopic = get_parameter("camera_pose_topic").as_string();

    // Initialize TF2 Buffer and TransformListener
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

    // Create publisher
    posePublisher = create_publisher<geometry_msgs::msg::TransformStamped>(poseTopic, 10);

    // Create timer for periodic publishing
    timer = create_wall_timer(
        std::chrono::duration<double>(1.0 / rateHz),
        std::bind(&CameraPosePublisher::publishPose, this));

    RCLCPP_INFO(get_logger(), "CameraPosePublisher started: %s → %s (%.1f Hz)",
                parentFrame.c_str(), childFrame.c_str(), rateHz);
}

void CameraPosePublisher::publishPose()
{
    geometry_msgs::msg::TransformStamped transform;

    try {
        transform = tfBuffer->lookupTransform(parentFrame, childFrame, tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", ex.what());
        return;
    }

    posePublisher->publish(transform);
}