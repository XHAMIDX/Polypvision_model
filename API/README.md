# Polyp Detection Chatbot with Streamlit

A medical AI application that analyzes polyp images and classifies them using OpenAI-compatible APIs.

## Features

- 🖼️ **Image Upload**: Upload polyp images for analysis
- 🤖 **AI-Powered Classification**: Identifies polyp types using vision models
- 📊 **Detailed Results**: Shows confidence levels and clinical recommendations
- 🔧 **Easy Configuration**: Uses environment variables for API settings

## Supported Polyp Types

1. **Hyperplastic** - Common, usually benign
2. **Adenomatous** - Requires removal, with subtypes:
   - **Tubular** - Most common subtype
   - **Villous** - Higher malignancy risk
   - **Tubulovillous** - Mixed characteristics

## Setup Instructions

### 1. Clone or Create Project Directory

```bash
mkdir polyp-detection
cd polyp-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Settings

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://agentrouter.org/v1
OPENAI_MODEL=gpt-5
```

**Important:** Replace the values with your actual credentials:
- `OPENAI_API_KEY`: Your API key from the OpenAI-compatible provider
- `OPENAI_BASE_URL`: The base URL for your API endpoint
- `OPENAI_MODEL`: The model name available on your service

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Usage

1. **Upload Image**: Click the upload button to select a polyp image
2. **Analyze**: Click the "🔍 Analyze Polyp" button
3. **View Results**: See the classification, confidence level, and recommendations

## Configuration

### Via .env File (Recommended)

Create `.env` file with your settings:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://your-api-endpoint.com/v1
OPENAI_MODEL=your-model-name
```

### Via Environment Variables

Set environment variables directly:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your_key_here"
$env:OPENAI_BASE_URL = "https://your-api-endpoint.com/v1"
$env:OPENAI_MODEL = "your-model-name"

# macOS/Linux
export OPENAI_API_KEY="your_key_here"
export OPENAI_BASE_URL="https://your-api-endpoint.com/v1"
export OPENAI_MODEL="your-model-name"
```

## Troubleshooting

### Error 401 - UNAUTHENTICATED
- **Cause**: Invalid API key or authentication failure
- **Solution**: 
  - Verify API key is correct in `.env` file
  - Check if API key has expired
  - Ensure API key has necessary permissions
  - Verify the API key format matches the provider's requirements

### Error 404 - NOT FOUND
- **Cause**: Incorrect base URL or model name
- **Solution**:
  - Double-check `OPENAI_BASE_URL` is correct
  - Verify model name is available on the service
  - Ensure the endpoint path is correct

### Connection Timeout
- **Cause**: Network issues or service unreachable
- **Solution**:
  - Check internet connection
  - Verify base URL is accessible
  - Check if the API service is online

## Testing the Connection

The application includes a built-in connection tester:

1. Open the app in browser
2. Click the **⚙️ Configuration** section in the sidebar
3. Click **🔧 Debug & Test** section
4. Click the **🧪 Test API Connection** button
5. Check the results for detailed error messages

## Project Structure

```
polyp-detection/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Requirements

- Python 3.8+
- pip package manager
- Internet connection for API calls
- Valid API credentials

## Dependencies

See `requirements.txt` for full list:

- streamlit: Web framework
- openai: OpenAI-compatible client library
- pillow: Image processing
- python-dotenv: Environment variable management
- requests: HTTP library

## API Compatibility

This application is compatible with any OpenAI-compatible API service, including:

- OpenAI's official API
- Azure OpenAI
- LocalAI
- Ollama
- Custom OpenAI-compatible providers

## Security Notes

⚠️ **Important**: 
- Never commit `.env` file to version control
- Keep your API key secret
- Use `.gitignore` to exclude `.env` from git

## Performance Tips

- Use clear, well-lit images for better accuracy
- Resize very large images before uploading
- Test API connection before analyzing images
- Check Debug section for detailed error information

## Support

For issues or questions:

1. Check the **Debug & Test** section in the sidebar
2. Review the troubleshooting section above
3. Verify all configuration settings
4. Check API service status

## License

This project is provided as-is for medical analysis purposes.

**Disclaimer**: This tool is for informational purposes only and should not replace professional medical diagnosis.

---

**Version**: 1.0  
**Last Updated**: November 2025
