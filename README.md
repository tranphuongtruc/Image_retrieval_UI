# Image Retrieval Project

## Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/tranphuongtruc/Image_retrieval_UI.git
    cd Image_retrieval_UI
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Requirements:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install Additional Dependencies from GitHub:**

    To install `transnet` and `CLIP` from GitHub, run the following commands:

    ```bash
    # Install TransNetV2
    pip install git+https://github.com/soCzech/TransNetV2.git

    # Install CLIP
    pip install git+https://github.com/openai/CLIP.git
    ```


5. **Run the Application:**

    ```bash
    python app.py
    ```

## Localtunnel

If you need to expose your local server to the internet, use Localtunnel:

1. **Install Localtunnel:**

    ```bash
    npm install -g localtunnel
    ```

2. **Run Localtunnel:**

    ```bash
    lt --port 5001
    ```

3. **Access the URL Provided by Localtunnel.**



## Setup and Run on Google Colab

### 1. Clone the Repository

Start by cloning the repository into your Google Colab environment:

```python
!git clone https://github.com/tranphuongtruc/Image_retrieval_UI.git
%cd Image_retrieval_UI
```

### 2. Install TransNetV2 and CLIP from GitHub and required dependencies
!pip install -r requirements.txt

!pip install git+https://github.com/tranphuongtruc/transnetv2.git
!pip install git+https://github.com/openai/CLIP.git
