# Line labels in sync file. Some of these are deprecated but are still here to be able to load old data. 
OPHYS_KEYS = ('2p_vsync', 'vsync_2p')

STIMULUS_KEYS = ('frames', 'stim_vsync', 'vsync_stim')
PHOTODIODE_KEYS = ('photodiode', 'stim_photodiode')
EYE_TRACKING_KEYS = ("eye_frame_received",  # Expected eye tracking
                                            # line label after 3/27/2020
                        # clocks eye tracking frame pulses (port 0, line 9)
                        "cam2_exposure",
                        # previous line label for eye tracking
                        # (prior to ~ Oct. 2018)
                        "eyetracking",
                        "eye_tracking",# An undocumented, but possible eye tracking line label  # NOQA E114
                        "eye_cam_frame_readout", #4/18/2025
                        "eye_cam_exposing", #4/18/2025
                        )  
BEHAVIOR_TRACKING_KEYS = ("beh_frame_received",  # Expected behavior line label after 3/27/2020  # NOQA E127
                                                 # clocks behavior tracking frame # NOQA E127
                                                 # pulses (port 0, line 8)
                            "cam1_exposure",
                            "behavior_monitoring",
                            "beh_cam_frame_readout", #4/18/2025
                            "beh_cam_exposing", # 4/18/2025
                            ) 

# Below line labels were added on 4/18/2025, based on sync file content                           
FACE_TRACKING_KEYS = ("face_cam_frame_readout",
                          "face_cam_exposing",
                          )

NOSE_TRACKING_KEYS = ("nose_cam_frame_readout",
                          "nose_cam_exposing",
                          )