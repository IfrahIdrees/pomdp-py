import os, sys


forgetfulness = 0 #0.05
RANDOM_BASELINE = False
question_title = "language-indexQuestionAsked"
explaset_title = "explaset-action"
feedback_title = "language-feedback"
baseline = False
''' Tracking engine variables'''
_no_trigger = None
_sleep_interval = None
_cond_satisfy = None
_cond_notsatisfy = None
_delete_trigger = None
_non_happen = None
_other_happen = None
_file_name = None
_output_file_name = None
Reward_output_file = "Reward"

