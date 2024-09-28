from pydantic import BaseModel
from fastapi.responses import JSONResponse
from modal import Image, App, web_endpoint
from typing import List, Optional

image = Image.debian_slim().pip_install("youtube_transcript_api")
app = App(name="youtube-transcript")

class TranscriptRequest(BaseModel):
    video_id: str
    languages: Optional[List[str]] = None

@app.function(image=image)
def get_youtube_transcript(video_id: str, languages: Optional[List[str]] = None):
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        if languages:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        else:
            return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

@app.function(image=image)
@web_endpoint(method="POST")
def get_transcript(request: TranscriptRequest):
    try:
        transcript = get_youtube_transcript.remote(request.video_id, request.languages)
        formatted_transcript = [f"{entry['start']}: {entry['text']}" for entry in transcript]
        return JSONResponse(content={"transcript": formatted_transcript})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    app.serve()