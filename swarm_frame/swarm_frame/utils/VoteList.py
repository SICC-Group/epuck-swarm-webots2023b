class VoteList:

    @staticmethod
    def update_opinion(opinion_list: list, opinion_msg, robot_id: int):
        """
        Adds the opinion message to the opinion list, if the ID is not the robot ID
        and not already contained.
        Updates the opinion of a stored message if a message with the same ID is already contained.

        :param opinion_list The list of opinions to update
        :param opinion_msg The message to add/update the list
        :param robot_id The id of the message receiving robot

        :return: The updated opinion list
        """
        if opinion_msg.id != robot_id:
            contained_flag = False
            for i in opinion_list:
                if i.id == opinion_msg.id:
                    # default: OpinionMessage
                    i.opinion = opinion_msg.opinion

                    contained_flag = True
                    break

            if not contained_flag:
                opinion_list.append(opinion_msg)

        if opinion_msg.id == robot_id:
            contained_flag = False
            for i in opinion_list:
                if i.id == opinion_msg.id:
                    # default: OpinionMessage
                    i.opinion = opinion_msg.opinion

                    contained_flag = True
                    break

            if not contained_flag:
                opinion_list.append(opinion_msg)
                
        return opinion_list
    
        


