import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from isaac_sim_msgs.srv._prompt import Prompt_Response
from PIL import Image
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from transformers import pipeline

from isaac_sim_msgs.srv import Prompt


class Instructor(Node):
    def __init__(self, model_name: str):
        super().__init__("instructor_node")
        self.get_logger().info("Instructor node has been started.")

        if model_name is None:
            self.get_logger().info("No model name provided. Exiting.")
            return

        device: torch.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        # self.model = pipeline(
        #         task="visual-question-answering",
        #         model=model_name,
        #         device=device,
        #         use_fast=False,
        #         trust_remote_code=True,
        #     )
        # self.model = pipeline(
        #     task="image-text-to-text",
        #     model="google/gemma-3-4b-pt",
        #     device=device,
        #     dtype=torch.bfloat16
        # )

        self.model = pipe = pipeline(
            "visual-question-answering", model=model_name, device=device, trust_remote_code=True
        )

        # self.get_logger().info(f"Loaded model: {model_name}")

        self.bridge = CvBridge()

        self.create_service(Prompt, "/Instructor/Prompt", self.prompt_callback)

    def prompt_callback(
        self, request: Prompt.Request, response: Prompt.Response
    ) -> Prompt_Response:

        # self.get_logger().info(f"Received prompt request: {request.prompt}")
        prompt: str = f"{request.prompt}"

        image = Image.fromarray(
            cv2.cvtColor(
                self.bridge.imgmsg_to_cv2(request.image, desired_encoding="bgr8"), cv2.COLOR_BGR2RGB
            )
        )

        result = self.model(
            image=image,
            question=prompt,
        )

        # result = self.model(image=image.convert('RGB'), question=prompt)

        self.get_logger().info(f"Prompt processed. Result: {result}")
        # response.response = " ".join([r['answer'] for r in result])
        response.response = result[0]["answer"]

        return response


def main(args=None):
    try:
        rclpy.init(args=args)

        executor = MultiThreadedExecutor(num_threads=32)
        # model = 'nvidia/Cosmos-Reason2-2B'
        # model = 'Salesforce/blip-vqa-base'
        # model = 'openbmb/MiniCPM-V-2'
        model = "/home/user/ros2_ws/blip-vqa-finetuned/checkpoint-1359"
        instructor_node = Instructor(model_name=model)

        executor.add_node(instructor_node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
