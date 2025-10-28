#include <rclcpp/rclcpp.hpp>
#include "openfusion_tf_bridge/openfusion_tf_bridge_node.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraPosePublisher>());
    rclcpp::shutdown();
    return 0;
}
