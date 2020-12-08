""" Viola jones parameters """
right_step = 1.01
right_size = 4

left_step = 1.01
left_size = 4

use_nose_detection = False


""" Paths """
path_to_ground_truth_bb = 'data/testannot_rect'
path_to_predicted_bb = 'data/test_viola_rect'
path_to_train_data = 'data/test'
path_to_annotated_data = 'data/testannot'

""" Evaluation """
iou_threshold = 0.5
