import gradio as gr

def positive_sample_training():
    return "Positive Sample Training"

def model_inference():
    return "Model Inference"

def zero_shot_online_learning_and_inference():
    return "Zero-Shot Online Learning and Inference"

def save_profile(avatar, name, email, organization, role, bio):
    if not name.strip():
        return "⚠️ 姓名不能为空 (Name cannot be empty)"
    if email.strip() and "@" not in email:
        return "⚠️ 邮箱格式不正确 (Invalid email format)"
    return (
        f"✅ 个人资料已保存 | Name: {name} | Email: {email} | "
        f"Organization: {organization} | Role: {role} | Bio: {bio}"
    )

with gr.Blocks(title="智能模型平台", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.Tab("主页 (Home)"):
            gr.Markdown("## 智能模型平台")
            with gr.Row():
                positive_training = gr.Button("正样本训练", variant="primary")
                inference = gr.Button("模型推理", variant="primary")
                zero_shot = gr.Button("零样本在线学习与推理", variant="primary")

            #positive_training.click(fn=positive_sample_training, inputs=[], outputs=[])
            #inference.click(fn=model_inference, inputs=[], outputs=[])
            #zero_shot.click(fn=zero_shot_online_learning_and_inference, inputs=[], outputs=[])

        with gr.Tab("用户资料 (Profile)"):
            gr.Markdown("## 用户资料 (User Profile)")
            with gr.Row():
                with gr.Column(scale=1):
                    profile_avatar = gr.Image(
                        value=None,
                        label="头像 (Avatar)",
                        height=150,
                        width=150,
                        type="pil",
                        interactive=True,
                    )
                with gr.Column(scale=2):
                    profile_name = gr.Textbox(
                        label="姓名 (Name)",
                        placeholder="请输入您的姓名 (Enter your name)",
                    )
                    profile_email = gr.Textbox(
                        label="邮箱 (Email)",
                        placeholder="请输入您的邮箱 (Enter your email)",
                    )
            with gr.Row():
                profile_organization = gr.Textbox(
                    label="单位 (Organization)",
                    placeholder="请输入您的单位 (Enter your organization)",
                )
                profile_role = gr.Textbox(
                    label="职位 (Role)",
                    placeholder="请输入您的职位 (Enter your role)",
                )
            profile_bio = gr.Textbox(
                label="个人简介 (Bio)",
                placeholder="请输入个人简介 (Enter a short bio)",
                lines=3,
            )
            save_btn = gr.Button("保存资料 (Save Profile)", variant="primary")
            save_status = gr.Textbox(label="状态 (Status)", interactive=False)

            save_btn.click(
                fn=save_profile,
                inputs=[profile_avatar, profile_name, profile_email, profile_organization, profile_role, profile_bio],
                outputs=save_status,
            )

    # 底部版权信息
    with gr.Row():
        footer = gr.Markdown("版权©2025 千瞳智能科技（苏州）有限公司. 所有权利保留")

demo.launch()
