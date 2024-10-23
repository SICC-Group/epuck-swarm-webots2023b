class PoseDict:

    @staticmethod
    def update_pose(pose_dict: dict, pose_msg):
        """
        Adds the pose message to the pose dictionary, if the ID is not the robot ID
        and not already contained.
        Updates the pose of a stored message if a message with the same ID is already contained.

        :param pose_dict The dictionary of pose to update
        :param pose_msg The message to add/update the dictionary
        :param robot_id The id of the message receiving robot

        :return: The updated pose dictionary
        """

        pose_dict[pose_msg.name] =  {"x": pose_msg.x, "y": pose_msg.y, "theta": pose_msg.theta}

        return pose_dict