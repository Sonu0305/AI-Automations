import os
import datetime
import re
import argparse
import speech_recognition as sr
from googletrans import Translator
from googleapiclient.discovery import build
from google.generativeai import configure, GenerativeModel

YOUTUBE_API_KEY = "PUT-YOUR-YOUTUBE_API_KEY"
GEMINI_API_KEY = "PUT-YOUR-GEMINI_API_KEY"

class YouTubeVideoFinder:
    def __init__(self):
        """Initialize the YouTube Video Finder with Gemini."""
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.translator = Translator()
        self.recognizer = sr.Recognizer()
        
        # Configure Gemini
        configure(api_key=GEMINI_API_KEY)
        self.model = GenerativeModel("gemini-1.5-flash")

    def get_input(self, voice_input=False):
        """Get input from user via text or voice."""
        if not voice_input:
            query = input("Enter your search query (Hindi/English): ")
        else:
            with sr.Microphone() as source:
                print("Speak your query (Hindi/English)...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
                
            try:
                try:
                    query = self.recognizer.recognize_google(audio, language="hi-IN")
                    print(f"Detected Hindi: {query}")
                except:
                    query = self.recognizer.recognize_google(audio, language="en-US")
                    print(f"Detected English: {query}")
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError:
                print("Could not request results from speech recognition service")
                return None
                
        detected = self.translator.detect(query)
        if detected.lang != 'en':
            query_en = self.translator.translate(query, dest='en').text
            print(f"Translated query: {query_en}")
            return query_en
        return query

    def search_youtube(self, query):
        """Search YouTube for videos matching the query with filtering."""
        two_weeks_ago = (datetime.datetime.now() - datetime.timedelta(days=50)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        search_response = self.youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=50,
            type='video',
            publishedAfter=two_weeks_ago,
            relevanceLanguage='en'
        ).execute()
        
        video_ids = [item['id']['videoId'] for item in search_response['items']]
        
        videos_response = self.youtube.videos().list(
            id=','.join(video_ids),
            part='contentDetails,statistics,snippet'
        ).execute()
        
        filtered_videos = []
        
        for video in videos_response['items']:
            duration = video['contentDetails']['duration']
            duration_seconds = self._parse_duration(duration)
            
            if 240 <= duration_seconds <= 1200:
                filtered_videos.append({
                    'id': video['id'],
                    'title': video['snippet']['title'],
                    'channelTitle': video['snippet']['channelTitle'],
                    'publishedAt': video['snippet']['publishedAt'],
                    'duration': duration,
                    'viewCount': video['statistics'].get('viewCount', '0'),
                    'url': f"https://www.youtube.com/watch?v={video['id']}"
                })
                
                if len(filtered_videos) >= 20:
                    break
        
        return filtered_videos

    def _parse_duration(self, duration_str):
        """Parse ISO 8601 duration format to seconds."""
        duration = duration_str[2:]
        seconds = 0
        
        if 'H' in duration:
            hours, duration = duration.split('H')
            seconds += int(hours) * 3600
        
        if 'M' in duration:
            minutes, duration = duration.split('M')
            seconds += int(minutes) * 60
        
        if 'S' in duration:
            s = duration.split('S')[0]
            seconds += int(s)
            
        return seconds

    def analyze_videos_with_llm(self, videos, query):
        """Analyze video titles using Gemini."""
        titles = [f"{i+1}. {video['title']} (by {video['channelTitle']})" for i, video in enumerate(videos)]
        titles_text = "\n".join(titles)
        
        prompt = f"""
        Analyze these YouTube video titles for their relevance to the query: "{query}"
        
        Videos:
        {titles_text}
        
        Based solely on the titles, which ONE video appears most relevant and high-quality for this query?
        Consider factors like: specificity to the query, informativeness, professional phrasing,
        lack of clickbait, and content quality signals.
        
        Return only the number of the best video with a brief explanation why it's the best match.
        Format: "Best video: [number] - [brief explanation]"
        """
        
        response = self.model.generate_content(prompt)
        return response.text

    def select_best_video(self, videos, analysis):
        """Extract the best video based on LLM analysis."""
        try:
            match = re.search(r"Best video:?\s*\[?(\d+)\]?", analysis)
            if match:
                best_index = int(match.group(1)) - 1
                if 0 <= best_index < len(videos):
                    return videos[best_index], analysis
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            
        return videos[0], "Could not parse LLM analysis. Defaulting to first result."

    def run(self, voice_input=False):
        """Run the complete workflow."""
        query = self.get_input(voice_input)
        if not query:
            return None, "No valid query provided"
        
        print(f"Searching YouTube for: {query}")
        videos = self.search_youtube(query)
        
        if not videos:
            return None, "No videos found matching the criteria"
        
        print(f"Found {len(videos)} videos matching criteria")
        
        print("Analyzing videos with Gemini...")
        analysis = self.analyze_videos_with_llm(videos, query)
        
        best_video, explanation = self.select_best_video(videos, analysis)
        
        return best_video, explanation

def main():
    parser = argparse.ArgumentParser(description='YouTube Video Finder with Gemini Analysis')
    parser.add_argument('--voice', action='store_true', help='Use voice input instead of text')
    args = parser.parse_args()
    
    if not YOUTUBE_API_KEY:
        print("ERROR: YouTube API key not found. Set the YOUTUBE_API_KEY environment variable.")
        return
        
    if not GEMINI_API_KEY:
        print("ERROR: Gemini API key not found. Set the GEMINI_API_KEY environment variable.")
        return
    
    finder = YouTubeVideoFinder()
    best_video, explanation = finder.run(voice_input=args.voice)
    
    if best_video:
        print("\n" + "="*50)
        print("BEST VIDEO FOUND:")
        print(f"Title: {best_video['title']}")
        print(f"Channel: {best_video['channelTitle']}")
        print(f"URL: {best_video['url']}")
        print(f"Published: {best_video['publishedAt']}")
        print("\nExplanation from Gemini:")
        print(explanation)
    else:
        print("\nNo suitable video found:", explanation)

if __name__ == "__main__":
    main()