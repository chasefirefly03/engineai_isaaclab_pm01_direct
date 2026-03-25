import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/pm01_dirct/usd_model/pm01/pm01.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
            # enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),    #0.8
        joint_pos={
            "j00_hip_pitch_l": -0.24,
            "j01_hip_roll_l": 0.0,
            "j02_hip_yaw_l": 0.0,
            "j03_knee_pitch_l": 0.48,
            "j04_ankle_pitch_l": -0.24,
            "j05_ankle_roll_l": 0.0,
            "j06_hip_pitch_r": -0.24,
            "j07_hip_roll_r": 0.0,
            "j08_hip_yaw_r": 0.0,
            "j09_knee_pitch_r": 0.48,
            "j10_ankle_pitch_r": -0.24,
            "j11_ankle_roll_r": 0.0,

            # 腰部 (1个关节)
            "j12_waist_yaw": 0.0,

            # 左臂 (5个关节)
            "j13_shoulder_pitch_l": 0.2, #0
            "j14_shoulder_roll_l": 0.13, #0.1
            "j15_shoulder_yaw_l": 0.0,
            "j16_elbow_pitch_l": -0.65, #0.0
            "j17_elbow_yaw_l": 0.0,

            # 右臂 (5个关节)
            "j18_shoulder_pitch_r": 0.2, #0
            "j19_shoulder_roll_r": -0.13, #0.1
            "j20_shoulder_yaw_r": 0.0,
            "j21_elbow_pitch_r": -0.65, #0.0
            "j22_elbow_yaw_r": 0.0,

            # 头部 (1个关节)
            "j23_head_yaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs":ImplicitActuatorCfg(
            joint_names_expr = [
                # 左腿 (6个关节)
                "j00_hip_pitch_l", "j01_hip_roll_l", "j02_hip_yaw_l", 
                "j03_knee_pitch_l", "j04_ankle_pitch_l", "j05_ankle_roll_l",
                
                # 右腿 (6个关节)
                "j06_hip_pitch_r", "j07_hip_roll_r", "j08_hip_yaw_r", 
                "j09_knee_pitch_r", "j10_ankle_pitch_r", "j11_ankle_roll_r",

                # 腰部 (1个关节)
                "j12_waist_yaw",
                
                # 左臂 (5个关节)
                "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l",
                "j16_elbow_pitch_l", "j17_elbow_yaw_l",
                
                # 右臂 (5个关节)
                "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r",
                "j21_elbow_pitch_r", "j22_elbow_yaw_r",
                
                # 头部 (1个关节)
                "j23_head_yaw"
            ],
            effort_limit={
                ".*_hip_pitch_.*": 164,
                ".*_hip_roll_.*": 164,
                ".*_hip_yaw_.*": 52,
                ".*_knee_pitch_.*": 164,
                ".*_ankle_pitch_.*": 52,
                ".*_ankle_roll_.*": 52,
                "j12_waist_yaw": 52,
                ".*_shoulder_.*": 52,
                ".*_elbow_.*": 52,
                "j23_head_yaw": 52,
            },

            velocity_limit={
                ".*_hip_pitch_.*": 26.3,
                ".*_hip_roll_.*": 26.3,
                ".*_hip_yaw_.*": 35.2,
                ".*_knee_pitch_.*": 26.3,
                ".*_ankle_pitch_.*": 35.2,
                ".*_ankle_roll_.*": 35.2,
                "j12_waist_yaw": 35.2,
                ".*_shoulder_.*": 35.2,
                ".*_elbow_.*": 35.2,
                "j23_head_yaw": 35.2,
            },

            stiffness={
                ".*_hip_roll_.*":   150,    
                ".*_hip_yaw_.*":    150,    

                ".*_hip_pitch_.*":  200,   
                ".*_knee_pitch_.*": 200,  
                "j12_waist_yaw":    200,

                ".*_ankle_.*":      150,
                
                ".*_shoulder_.*":   80,
                ".*_elbow_.*":      80,
                "j23_head_yaw":     80,      
            },
            # stiffness={
            #     ".*_hip_roll_.*":   150,    
            #     ".*_hip_yaw_.*":    150,    

            #     ".*_hip_pitch_.*":  200,   
            #     ".*_knee_pitch_.*": 200,  
            #     "j12_waist_yaw":    200,

            #     ".*_ankle_.*":      50,
                
            #     ".*_shoulder_.*":   40,
            #     ".*_elbow_.*":      40,
            #     "j23_head_yaw":     40,      
            # },            
            damping={
                ".*_hip_pitch_.*":  7.0,   
                ".*_hip_roll_.*":   3.0,   
                ".*_hip_yaw_.*":    3.0,     
                ".*_knee_pitch_.*": 7.0,  
                ".*_ankle_.*":      7.0,
                "j12_waist_yaw":    3.0,

                ".*_shoulder_.*":   3.0,
                ".*_elbow_.*":      3.0,
                "j23_head_yaw":     3.0,  
            },  
            armature = {
            ".*_hip_pitch_.*": 0.0453,
            ".*_hip_roll_.*": 0.0453,
            ".*_hip_yaw_.*": 0.0067,
            ".*_knee_.*": 0.0453,
            ".*_ankle_.*": 0.0067,
            ".*_waist_.*": 0.0067,
            ".*_shoulder_.*": 0.0067,
            ".*_elbow_.*": 0.0067,
            ".*_head_.*": 0.0067,
        }
        ),

    },
)
