# Use official lightweight Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the repository files to the container
COPY . /app

# Install CPU-only PyTorch first (to keep image small)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the pinagmm package and its dependencies
RUN pip install .

# Set environment variables so the app binds to 0.0.0.0 (required for Hugging Face)
ENV PORT=7860
ENV HOST="0.0.0.0"
ENV SHOW="false"

# Expose the Hugging Face Spaces port
EXPOSE 7860

# Command to run the application
CMD ["pinagmm"]
