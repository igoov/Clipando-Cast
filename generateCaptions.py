import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import whisper
import argparse
import textwrap
from typing import List, Dict, Any, Tuple
from collections import deque

def extract_audio(video_path, output_path="temp_audio.wav"):
    """Extract audio from video file"""
    command = f'ffmpeg -i "{video_path}" -ab 160k -ac 2 -ar 44100 -vn "{output_path}" -y'
    subprocess.call(command, shell=True)
    return output_path


def transcribe_audio(audio_path, whisper_model_size="base"):
    """Transcribe audio using Whisper and return segments"""
    print("Loading Whisper model...")
    model = whisper.load_model(whisper_model_size)
    
    print(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # Extract segments from the result
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": segment.get("words", [])
        })
    
    return segments


def review_transcription(transcription_segments):
    """Allow user to review and edit the transcription"""
    print("\n=== Transcription Review ===")
    print("Review each segment and correct any errors.")
    print("For each segment, you can:")
    print("  [enter] - Accept as is")
    print("  [edit text] - Type corrected text")
    print("  q - Finish review and save changes")
    print("  s - Skip to end without further review\n")
    
    for i, segment in enumerate(transcription_segments):
        print(f"\nSegment {i+1}/{len(transcription_segments)}")
        print(f"[{format_time(segment['start'])} - {format_time(segment['end'])}]")
        print(f"Current text: {segment['text']}")
        
        user_input = input("Correction (or enter to accept): ")
        
        if user_input.lower() == 'q':
            print("Finishing review...")
            break
        elif user_input.lower() == 's':
            print("Skipping remaining segments...")
            break
        elif user_input:
            # Update the segment text with correction
            transcription_segments[i]['text'] = user_input
            print(f"Updated: {user_input}")
            
            # Update words if they exist (simple approach - just reset timing)
            if 'words' in segment and segment['words']:
                avg_word_duration = (segment['end'] - segment['start']) / len(user_input.split())
                new_words = []
                words = user_input.split()
                current_time = segment['start']
                
                for word in words:
                    word_end = current_time + avg_word_duration
                    new_words.append({
                        "word": word,
                        "start": current_time,
                        "end": word_end
                    })
                    current_time = word_end
                
                transcription_segments[i]['words'] = new_words
    
    print("\nTranscription review complete!")
    return transcription_segments


def process_segments_for_captions(segments, video_width):
    """Keep word timing data, no line grouping.

    Older versions built long lines which made the TikTok style impossible.
    Here we just return the segments unmodified; the renderer uses the
    individual words directly.
    """
    # nothing to do other than ensure every segment has a word list
    for segment in segments:
        if "words" not in segment:
            segment["words"] = []
    return segments


def cv2_to_pil(cv2_img):
    """Convert CV2 image (BGR) to PIL image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    """Convert PIL image (RGB) to CV2 image (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_rounded_rectangle(draw, bbox, radius, fill):
    """Draw a rounded rectangle"""
    x1, y1, x2, y2 = bbox
    
    # Fix: Ensure x2 > x1 and y2 > y1
    if x2 <= x1 or y2 <= y1:
        # Invalid rectangle, skip drawing
        return
    
    # Ensure radius isn't too large for the rectangle
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius <= 0:
        # If radius is invalid, just draw a normal rectangle
        draw.rectangle((x1, y1, x2, y2), fill=fill)
        return
        
    draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill)
    draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill)
    # Draw four corners
    draw.pieslice((x1, y1, x1 + radius * 2, y1 + radius * 2), 180, 270, fill=fill)
    draw.pieslice((x2 - radius * 2, y1, x2, y1 + radius * 2), 270, 360, fill=fill)
    draw.pieslice((x1, y2 - radius * 2, x1 + radius * 2, y2), 90, 180, fill=fill)
    draw.pieslice((x2 - radius * 2, y2 - radius * 2, x2, y2), 0, 90, fill=fill)


def format_time(seconds):
    """Format seconds to mm:ss format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def create_blurred_background(frame, x1, y1, x2, y2, blur_amount=15):
    """Create a blurred background from a region of the frame"""
    # Extract the region to blur
    region = frame[y1:y2, x1:x2]
    
    # Apply blur
    blurred = cv2.GaussianBlur(region, (blur_amount, blur_amount), 0)
    
    # Return blurred region
    return blurred


def parse_color(color_str):
    """Parse color string in format 'r,g,b,a' or 'r,g,b'"""
    parts = color_str.split(',')
    if len(parts) == 3:
        # RGB format
        return tuple(int(p.strip()) for p in parts) + (255,)  # Default alpha to 255
    elif len(parts) == 4:
        # RGBA format
        return tuple(int(p.strip()) for p in parts)
    else:
        raise ValueError("Color must be in format 'r,g,b' or 'r,g,b,a'")


def find_current_word(segments, t):
    """Return the word dictionary active at time t, or None."""
    for seg in segments:
        for w in seg.get("words", []):
            if w["start"] <= t <= w["end"]:
                return w
    return None


def caption_video(video_path, output_path, segments, bg_color=(255, 255, 255, 0), 
                 highlight_color=(255, 226, 165, 220), text_color=(0, 0, 0)):
    """Add TikTok‑style single‑word captions to the video.

    Words are rendered one at a time, bright yellow with a black outline and a
    brief scale "pulse" when they appear. The original multi‑line machinery has
    been removed for simplicity.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(f"{output_path}_temp.mp4", fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{output_path}_temp.mp4", fourcc, fps, (width, height))
        if not out.isOpened():
            print("Could not open video writer with either codec")
            return None

    # large font for TikTok-style, minimum ~250px
    base_font_size = max(250, int(width * 0.25))
    pulse_duration = 0.3

    frame_idx = 0
    print(f"Processing {total_frames} frames at {fps} fps")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames")

        word = find_current_word(segments, t)
        if word:
            print(f"[debug] generating caption word '{word['text']}' at {t:.2f}s")
            age = t - word["start"]
            scale = 1.0
            if 0 <= age < pulse_duration:
                scale += 0.2 * np.sin(np.pi * age / pulse_duration)

            txt = word["text"].strip()
            # adjust font size for long words
            font_size = base_font_size
            if len(txt) > 10:
                font_size = max(100, base_font_size - (len(txt) - 10) * 5)

            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(font_size * scale))
            except Exception:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0,0), txt, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = (width - w) / 2
            y = height - h - 100
            draw.text((x, y), txt, font=font, fill="#FFD700", stroke_width=8, stroke_fill="black")
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed {frame_idx} frames")

    final_output = f"{output_path}"
    combine_cmd = f'ffmpeg -i "{output_path}_temp.mp4" -i "{video_path}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{final_output}" -y'
    subprocess.call(combine_cmd, shell=True)
    if os.path.exists(f"{output_path}_temp.mp4"):
        os.remove(f"{output_path}_temp.mp4")
    if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
        print(f"Successfully created captioned video at {final_output}")
        return final_output
    else:
        print(f"Failed to create captioned video at {final_output}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Caption an entire video with word-level subtitles")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-path", help="Path for the output video file", default="captioned_video.mp4")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use for transcription")
    parser.add_argument("--transcription-path", help="Path to existing transcription JSON (optional)")
    parser.add_argument("--skip-review", action="store_true", help="Skip transcription review")
    
    # Add color customization options
    parser.add_argument("--bg-color", default="255,255,255,0", 
                        help="Background color in format 'r,g,b,a' (e.g., '255,255,255,0')")
    parser.add_argument("--highlight-color", default="255,226,165,220", 
                        help="Highlight color in format 'r,g,b,a' (e.g., '255,226,165,220')")
    parser.add_argument("--text-color", default="0,0,0", 
                        help="Text color in format 'r,g,b' (e.g., '0,0,0')")
    
    args = parser.parse_args()
    
    # Parse color arguments
    try:
        bg_color = parse_color(args.bg_color)
        highlight_color = parse_color(args.highlight_color)
        text_color = parse_color(args.text_color)[:3]  # Text color doesn't need alpha
    except ValueError as e:
        print(f"Error parsing color: {e}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load existing transcription or create new one
    if args.transcription_path and os.path.exists(args.transcription_path):
        print(f"Loading existing transcription from {args.transcription_path}")
        with open(args.transcription_path, "r") as f:
            transcription_segments = json.load(f)
    else:
        # Extract audio from video
        print("Extracting audio from video...")
        audio_path = extract_audio(args.video_path)
        
        # Transcribe audio
        print("Transcribing audio...")
        transcription_segments = transcribe_audio(audio_path, args.whisper_model)
        
        # Save original transcription to file
        original_transcription_path = os.path.join(os.path.dirname(args.output_path), "original_transcription.json")
        try:
            with open(original_transcription_path, "w", encoding="utf-8") as f:
                json.dump(transcription_segments, f, indent=2)
            print(f"Original transcription saved to {original_transcription_path}")
        except IOError as e:
            print(f"Failed to save original transcription: {e}")
        
        # Clean up audio file
        os.remove(audio_path)
    
    # Review transcription if not skipped
    if not args.skip_review:
        print("\nReviewing transcription...")
        transcription_segments = review_transcription(transcription_segments)
        
        # Save reviewed transcription
        reviewed_transcription_path = os.path.join(os.path.dirname(args.output_path), "reviewed_transcription.json")
        try:
            with open(reviewed_transcription_path, "w", encoding="utf-8") as f:
                json.dump(transcription_segments, f, indent=2)
            print(f"Reviewed transcription saved to {reviewed_transcription_path}")
        except IOError as e:
            print(f"Failed to save reviewed transcription: {e}")
    
    # Get video info to calculate dimensions for captions
    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    # Process segments for captions
    print("Processing segments for captioning...")
    transcription_segments = process_segments_for_captions(transcription_segments, width)
    
    # Caption the video with custom colors
    print("Adding captions to video...")
    captioned_video = caption_video(
        args.video_path, 
        args.output_path, 
        transcription_segments,
        bg_color=bg_color,
        highlight_color=highlight_color,
        text_color=text_color
    )
    
    if captioned_video:
        print(f"\nProcess complete! Captioned video saved to {captioned_video}")
        # Print the color settings used
        print(f"Caption settings:")
        print(f"  Background color: {bg_color}")
        print(f"  Highlight color: {highlight_color}")
        print(f"  Text color: {text_color}")
    else:
        print("\nFailed to create captioned video.")


if __name__ == "__main__":
    main()