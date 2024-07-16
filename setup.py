from setuptools import setup, find_packages

setup(
    name="webrtc_yolo_app",
    version="0.1.0",
    description="A WebRTC application with YOLO for object detection using Streamlit",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/your-repo",  # Update this with your repository URL
    packages=find_packages(),
    install_requires=[
        "opencv-python-headless",
        "streamlit-webrtc",
        "av",
        "numpy",
        "streamlit",
        "pytest",
        "selenium"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
