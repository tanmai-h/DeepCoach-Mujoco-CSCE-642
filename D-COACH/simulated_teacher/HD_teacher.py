from tools.functions import observation_to_gray
from autoencoder import AE
from simulated_teacher.teacher_base import TeacherBase
import cv2


class Teacher(TeacherBase):
    def __init__(self, image_size=64, dim_a=3, action_lower_limits='0,0,0', action_upper_limits='1,1,1',
                 loc='graphs/teacher/CarRacing-v0/network', error_prob=0, resize_observation=True,
                 teacher_parameters='0.6,0.00001', pickFetchEnv=None):

        super(Teacher, self).__init__(dim_a=dim_a, action_lower_limits=action_lower_limits,
                                      action_upper_limits=action_upper_limits, loc=loc,
                                      error_prob=error_prob, teacher_parameters=teacher_parameters, pickFetchEnv=pickFetchEnv)

    def _preprocess_observation(self, observation):
        pass

