# Tested
def load_hvc_model(model_path, neural_network, num_classes, drop_off, total_sub_models_num, num_vc, p_h, p_w):
    """
    load the custom heretical visual concept model
    """
    sub_model_num = int(model_path[-4:-3])
    print(f'sub_model_num = {sub_model_num}')
    # Create the sub_model
    hvc_sub_mode, _, _, _ = neural_network(num_classes=num_classes,
                                           dropout_rate=drop_off,
                                           num_vc=num_vc,
                                           layer_to_vcl=sub_model_num,
                                           p_h=p_h,
                                           p_w=p_w,
                                           total_sub_models_num=total_sub_models_num)
    hvc_sub_mode.load_weights(model_path)
    return hvc_sub_mode
