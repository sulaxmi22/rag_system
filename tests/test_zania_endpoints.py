"""
Tests for file upload, question processing, and batch Q&A functionality
"""

import pytest
import json
import tempfile
import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def sample_json_document():
    """Create a sample JSON document for testing"""
    return {
        "content": "This is a test document. It contains information about security policies. Our company has a 24-hour notification SLA for security incidents. We use AWS and Google Cloud for hosting."
    }


@pytest.fixture
def sample_questions():
    """Create sample questions for testing"""
    return {
        "questions": [
            "What is the notification SLA for security incidents?",
            "Which cloud providers do you use?"
        ]
    }


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_upload_json_document(sample_json_document):
    """Test uploading a JSON document"""
    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_json_document, f)
        temp_file_path = f.name
    
    try:
        # Upload the file
        with open(temp_file_path, 'rb') as f:
            response = client.post(
                "/upload/document",
                files={"file": ("test.json", f, "application/json")},
                data={"source": "test"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "doc_id" in data
        assert data["file_type"] == ".json"
    finally:
        # Clean up
        os.unlink(temp_file_path)


def test_upload_questions(sample_questions):
    """Test uploading a JSON questions file"""
    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_questions, f)
        temp_file_path = f.name
    
    try:
        # Upload the file
        with open(temp_file_path, 'rb') as f:
            response = client.post(
                "/upload/questions",
                files={"file": ("questions.json", f, "application/json")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["question_count"] == 2
    finally:
        # Clean up
        os.unlink(temp_file_path)


def test_upload_questions_invalid_json():
    """Test uploading invalid JSON file"""
    # Create temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json content")
        temp_file_path = f.name
    
    try:
        # Upload the file
        with open(temp_file_path, 'rb') as f:
            response = client.post(
                "/upload/questions",
                files={"file": ("questions.json", f, "application/json")}
            )
        
        assert response.status_code == 400
    finally:
        # Clean up
        os.unlink(temp_file_path)


def test_upload_document_unsupported_type():
    """Test uploading unsupported file type"""
    # Create temporary text file (unsupported)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a text file")
        temp_file_path = f.name
    
    try:
        # Upload the file
        with open(temp_file_path, 'rb') as f:
            response = client.post(
                "/upload/document",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    finally:
        # Clean up
        os.unlink(temp_file_path)


def test_answer_questions_without_upload():
    """Test answering questions without uploading questions first"""
    response = client.post("/answer-questions")
    assert response.status_code == 400
    assert "No questions uploaded" in response.json()["detail"]


def test_stats_endpoint():
    """Test statistics endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "routing_stats" in data
    assert "cache_stats" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
