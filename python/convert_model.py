# # 필요한 라이브러리 임포트
# # pip install tensorflow==1.15.0
# # pip install protobuf==3.20.3
# # pip install tf2onnx
# # pip install onnx onnx2pytorch

# import tensorflow as tf
# from tensorflow.python.framework import graph_util
# import tf2onnx
# import onnx
# from onnx2pytorch import ConvertModel
# import torch
# import os

# # TensorFlow 1.x 호환성 모드 활성화
# tf.compat.v1.disable_v2_behavior()

# # TensorFlow 모델 변환 함수 정의
# def freeze_graph(ckpt_path, output_node_names, pb_path):
#     with tf.Session() as sess:
#         # 그래프 복원
#         saver = tf.train.import_meta_graph(ckpt_path + '.meta')
#         saver.restore(sess, ckpt_path)

#         # 그래프 정의 가져오기
#         graph_def = sess.graph.as_graph_def()

#         # 변수들을 상수로 변환하여 그래프 프리즈
#         frozen_graph_def = graph_util.convert_variables_to_constants(
#             sess, graph_def, output_node_names)

#         # 프리즈된 그래프를 .pb 파일로 저장
#         with open(pb_path, 'wb') as f:
#             f.write(frozen_graph_def.SerializeToString())
#         print(f"Frozen graph saved at: {pb_path}")

# def convert_to_onnx(pb_path, onnx_path, input_names, output_names):
#     # tf2onnx를 사용하여 변환
#     command = f'python -m tf2onnx.convert --input {pb_path} --output {onnx_path} ' \
#               f'--inputs {",".join(input_names)} --outputs {",".join(output_names)}'
#     os.system(command)
#     print(f"ONNX model saved at: {onnx_path}")

# def convert_to_pytorch(onnx_path, pt_path):
#     onnx_model = onnx.load(onnx_path)
#     pytorch_model = ConvertModel(onnx_model)
#     torch.save(pytorch_model.state_dict(), pt_path)
#     print(f"PyTorch model saved at: {pt_path}")

# # # 메인 실행 함수
# # if __name__ == "__main__":
# #     # 변환할 체크포인트 파일 목록 설정
# #     ckpt_dir = './ckpt_files'  # 체크포인트 파일이 있는 디렉터리로 변경하세요
# #     ckpt_files = [
# #         os.path.join(ckpt_dir, f"model_badmode2_run{run}.ckpt") for run in range(15, 20)
# #     ] + [
# #         os.path.join(ckpt_dir, f"model_badmode4_run{run}.ckpt") for run in range(15, 20)
# #     ]

# #     # 출력 노드 및 입력 노드 이름 설정
# #     output_node_names = ['q_p1', 'q_0']
# #     input_names = ['input_0:0', 'input_1:0', 'eps:0']
# #     output_names = [name + ':0' for name in output_node_names]

# #     for ckpt_path in ckpt_files:
# #         # 단계별 파일 경로 설정
# #         base_name = os.path.basename(ckpt_path).replace(".ckpt", "")
# #         pb_path = f"{base_name}_frozen_model.pb"
# #         onnx_path = f"{base_name}.onnx"
# #         pt_path = f"{base_name}.pt"

# #         # 1. 그래프 프리즈하여 .pb 파일 생성
# #         freeze_graph(ckpt_path, output_node_names, pb_path)

# #         # 2. .pb 파일을 ONNX 형식으로 변환
# #         convert_to_onnx(pb_path, onnx_path, input_names, output_names)

# #         # 3. ONNX 모델을 PyTorch 모델로 변환하여 .pt 파일로 저장
# #         convert_to_pytorch(onnx_path, pt_path)

# # # import tensorflow as tf

# # # # 예시: 특정 체크포인트의 .meta 파일 경로를 입력하세요.
# # # ckpt_path = './ckpt/model_badmode2_run15.ckpt'

# # # with tf.compat.v1.Session() as sess:
# # #     saver = tf.compat.v1.train.import_meta_graph(ckpt_path + '.meta')
# # #     saver.restore(sess, ckpt_path)
# # #     print("모든 노드 이름을 출력합니다:")
# # #     for node in sess.graph.as_graph_def().node:
# # #         print(node.name)

# # 수정된 메인 함수
# if __name__ == "__main__":
#     # 변환할 체크포인트 파일 목록 설정
#     ckpt_dir = './ckpt_files'  # 올바른 디렉터리 설정
#     ckpt_files = [
#         os.path.join(ckpt_dir, f"model_badmode2_run{run}.ckpt") for run in range(15, 20)
#     ] + [
#         os.path.join(ckpt_dir, f"model_badmode4_run{run}.ckpt") for run in range(15, 20)
#     ]

#     # 출력 노드 및 입력 노드 이름 (확인 후 수정)
#     output_node_names = ['q_p1', 'q_0']
#     input_names = ['input_0:0', 'input_1:0', 'eps:0']
#     output_names = [name + ':0' for name in output_node_names]

#     for ckpt_path in ckpt_files:
#         base_name = os.path.basename(ckpt_path).replace(".ckpt", "")
#         pb_path = f"{base_name}_frozen_model.pb"
#         onnx_path = f"{base_name}.onnx"
#         pt_path = f"{base_name}.pt"

#         # 1. 그래프 프리즈하여 .pb 파일 생성
#         try:
#             freeze_graph(ckpt_path, output_node_names, pb_path)
#         except Exception as e:
#             print(f"Error during freezing graph: {e}")
#             continue

#         # 2. .pb 파일을 ONNX 형식으로 변환
#         try:
#             convert_to_onnx(pb_path, onnx_path, input_names, output_names)
#         except Exception as e:
#             print(f"Error during ONNX conversion: {e}")
#             continue

#         # 3. ONNX 모델을 PyTorch 모델로 변환
#         try:
#             convert_to_pytorch(onnx_path, pt_path)
#         except Exception as e:
#             print(f"Error during PyTorch conversion: {e}")


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 저장된 체크포인트 경로를 설정합니다.
ckpt_path = './ckpt_files/model_badmode2_run15.ckpt'

# 세션에서 그래프를 로드하고 노드 이름을 출력합니다.
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')  # .meta 파일 로드
    saver.restore(sess, ckpt_path)  # 체크포인트 로드

    print("그래프에 정의된 모든 노드 이름:")
    for node in sess.graph.as_graph_def().node:
        print(node.name)
