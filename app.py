#!/usr/bin/env python3

import os
import sys
import platform
import json
import uuid
import threading
import subprocess
import socket
from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB


@app.after_request
def add_local_dev_cors_headers(response):
    """Ensure local-dev requests always get CORS headers."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    # Avoid stale cached HTML/inline JS during local dev troubleshooting.
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/favicon.ico')
def favicon():
    # Silence browser favicon 404 noise in local dev.
    return Response(status=204)


def _find_executable(name):
    # Search sys.path for executable, with Windows .exe fallback
    if os.path.isabs(name) and os.path.exists(name):
        return name
    for path_dir in os.environ.get('PATH', '').split(os.pathsep):
        full = os.path.join(path_dir, name)
        if os.path.exists(full) and os.access(full, os.X_OK):
            return full
        if platform.system() == 'Windows':
            fullexe = full + '.exe'
            if os.path.exists(fullexe) and os.access(fullexe, os.X_OK):
                return fullexe
    return None


def ffmpeg_cmd():
    return os.environ.get('FFMPEG_PATH') or _find_executable('ffmpeg') or 'ffmpeg'


def ffprobe_cmd():
    return os.environ.get('FFPROBE_PATH') or _find_executable('ffprobe') or 'ffprobe'


def user_ffmpeg_install_hint():
    system = platform.system()
    if system == 'Windows':
        return 'Install ffmpeg from https://ffmpeg.org/download.html and add the bin folder to PATH.'
    if system == 'Darwin':
        return 'Install ffmpeg with Homebrew: brew install ffmpeg.'
    return 'Install ffmpeg via your package manager, e.g., apt install ffmpeg on Debian/Ubuntu.'


def is_port_in_use(host: str, port: int) -> bool:
    """Return True when a local TCP port is already bound by another process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# In-memory conversation history storage (job_id -> conversation)
conversation_histories = {}

# In-memory processing jobs status (job_id -> status dict)
processing_jobs = {}


SYSTEM_PROMPT = """You are a datamoshing video effects expert. Given a description of a desired glitch effect,
return a JSON object with the datamoshing parameters.

Two effects are available:
1. "iframe_removal" — removes i-frames (keyframes) between start and end, creating a glitchy transition where
   the background freezes and foreground motion bleeds through. Good for: transitions, scene changes, glitch art.

2. "delta_repeat" — captures a sequence of motion delta frames and loops them, creating a melting/smearing distortion.
   Good for: melting effects, psychedelic motion trails, warping visuals.

Parameters to return:
{
  "effect": "iframe_removal" | "delta_repeat",
  "start_frame": integer (default 0),
  "end_frame": integer (-1 means until end of video, default -1),
  "fps": integer (prefer the source clip fps when provided, otherwise default 30, typical range 24-60),
  "delta": integer (only for delta_repeat — number of frames to loop, default 5, best range 3-15),
  "explanation": "one sentence describing what will happen visually"
}

Interpret vague phrases generously:
- "transition", "scene change", "glitchy entrance/exit" → iframe_removal
- "melting", "smearing", "warping", "trails", "psychedelic" → delta_repeat
- "beginning / start" → start_frame ~0
- "middle" → center the effect around the middle of the actual clip
- "end" → use the last part of the actual clip and end_frame -1
- If no frame numbers given, use a reasonable range (e.g. 0 to -1 for full video)."""

NARRATIVE_SYSTEM_PROMPT = """You are a whimsical, imaginative storyteller—like an ELIZA therapist crossed with a dream interpreter. 
A user has described or shown you a video, and you weave their words and the video's essence into an evocative narrative.

Your style:
- Dreamlike, poetic, introspective
- Use vivid sensory language
- Build on their input to expand the narrative
- Maintain the spirit of their original description
- Respond with creative imagery and metaphysical wonder
- Keep responses concise (2-3 sentences) for immersion

You are helping them explore the emotional or symbolic meaning of what they've shared."""


def get_narrative_response(user_query: str, conversation_history: list, api_key: str) -> tuple:
    """Generate a narrative response and maintain conversation history."""
    try:
        client = OpenAI(api_key=api_key)
        # Append the new user message to the conversation history
        conversation_history.append({"role": "user", "content": user_query})

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=250
        )
        ai_response_content = completion.choices[0].message.content
        # Append the AI's response to the conversation history
        conversation_history.append({"role": "assistant", "content": ai_response_content})

        return ai_response_content, conversation_history
    except Exception as e:
        return f"An error occurred: {e}", conversation_history


def generate_image(prompt: str, api_key: str) -> str:
    """Generate an image using DALL-E based on a textual prompt."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        return f"Error generating image: {str(e)}"


def describe_video_content(video: str, api_key: str) -> str:
    """Get AI to describe what it sees in a video (using vision understanding)."""
    try:
        client = OpenAI(api_key=api_key)
        # For now, return a placeholder. In production, you'd use video_file parameter
        # when OpenAI supports full video analysis
        return "Video uploaded and ready for narrative generation."
    except Exception as e:
        return f"Error describing video: {str(e)}"



def parse_fraction(value: str) -> float:
    if not value or value == '0/0':
        return 0.0

    if '/' in value:
        numerator, denominator = value.split('/', 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value

    return float(value)


def get_video_metadata(input_path: str) -> dict:
    try:
        result = subprocess.run(
            [
                ffprobe_cmd(), '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate,r_frame_rate,nb_frames,duration',
                '-show_entries', 'format=duration',
                '-of', 'json',
                input_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout or '{}')
        stream = (data.get('streams') or [{}])[0]
        format_info = data.get('format') or {}

        duration = float(stream.get('duration') or format_info.get('duration') or 0)
        fps = parse_fraction(stream.get('avg_frame_rate') or stream.get('r_frame_rate') or '0')
        if fps <= 0:
            fps = 30.0

        nb_frames = stream.get('nb_frames')
        total_frames = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else int(round(duration * fps))
        total_frames = max(total_frames, 1)

        return {
            'duration_seconds': round(duration, 2),
            'fps': round(fps, 2),
            'total_frames': total_frames
        }
    except Exception:
        return {
            'duration_seconds': 0,
            'fps': 30.0,
            'total_frames': 300
        }


def normalize_ai_params(params: dict, video_metadata: dict) -> dict:
    total_frames = max(1, int(video_metadata.get('total_frames', 300)))
    source_fps = float(video_metadata.get('fps', 30))

    effect = params.get('effect', 'iframe_removal')
    if effect not in {'iframe_removal', 'delta_repeat'}:
        effect = 'iframe_removal'

    start_frame = max(0, int(params.get('start_frame', 0)))
    end_frame = int(params.get('end_frame', -1))

    if start_frame >= total_frames:
        start_frame = max(0, total_frames - 1)

    if end_frame != -1:
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

    fps = int(round(float(params.get('fps', source_fps or 30))))
    fps = max(1, min(120, fps))

    delta = int(params.get('delta', 5))
    delta = max(1, min(delta, 30))

    return {
        'effect': effect,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'fps': fps,
        'delta': delta,
        'explanation': params.get('explanation', '')
    }


def build_ai_user_prompt(prompt: str, video_metadata: dict) -> str:
    duration = video_metadata.get('duration_seconds', 0)
    fps = video_metadata.get('fps', 30)
    total_frames = video_metadata.get('total_frames', 300)

    return (
        "Video context:\n"
        f"- duration_seconds: {duration}\n"
        f"- source_fps: {fps}\n"
        f"- total_frames: {total_frames}\n"
        "- Use these values when interpreting words like beginning, middle, end, short, brief, or full video.\n"
        "- Keep start_frame and end_frame within the clip. Use end_frame = -1 only when the effect should run until the end.\n"
        "- Prefer the source_fps unless the user clearly asks for something else.\n\n"
        f"User request:\n{prompt}"
    )


def parse_prompt_with_ai(prompt: str, api_key: str, video_metadata: dict) -> dict:
    try:
        client = OpenAI(api_key=api_key)

        # OpenAI SDK has varying wrappers. The goal is to get a JSON-formatted model output.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_ai_user_prompt(prompt, video_metadata)}
            ],
            temperature=0.3,
            max_tokens=250
        )

        # Try to parse content from typical SDK response shapes.
        content = None
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if isinstance(choice, dict):
                content = choice.get('message', {}).get('content') if choice.get('message') else choice.get('text')
            else:
                content = getattr(choice, 'message', None)
                if content and hasattr(content, 'content'):
                    content = content.content
                elif content and isinstance(content, dict):
                    content = content.get('content')
        if not content and isinstance(response, dict):
            content = (response.get('choices') or [{}])[0].get('message', {}).get('content')

        if not content:
            raise ValueError('No text returned from OpenAI completion')

        content = content.strip()
        parsed = json.loads(content)

        return normalize_ai_params(parsed, video_metadata)
    except json.JSONDecodeError as jde:
        return {"error": f"AI response is not valid JSON: {jde}. Raw output: {content!r}"}
    except Exception as e:
        return {"error": str(e)}


def run_mosh(input_path: str, output_path: str, params: dict, job_id: str) -> dict:
    effect = params.get('effect', 'iframe_removal')
    start_frame = int(params.get('start_frame', 0))
    end_frame = int(params.get('end_frame', -1))
    fps = max(1, min(120, int(params.get('fps', 30))))
    delta = int(params.get('delta', 5)) if effect == 'delta_repeat' else 0

    input_avi = f'tmp_{job_id}_in.avi'
    output_avi = f'tmp_{job_id}_out.avi'

    try:
        # Step 1: convert input to AVI (required for frame byte manipulation)
        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-i', input_path,
            '-crf', '0', '-pix_fmt', 'yuv420p', '-bf', '0',
            '-b', '10000k', '-r', str(fps),
            input_avi
        ])
        if ret != 0 or not os.path.exists(input_avi):
            return {'success': False, 'error': 'ffmpeg AVI conversion failed. Is ffmpeg installed?'}

        # Step 2: read raw bytes and split into frames
        with open(input_avi, 'rb') as f:
            data = f.read()

        frame_marker = bytes.fromhex('30306463')  # '00dc' — end-of-frame marker
        iframe_sig = bytes.fromhex('0001B0')       # MPEG i-frame signature
        pframe_sig = bytes.fromhex('0001B6')       # MPEG p-frame (delta) signature

        frames = data.split(frame_marker)
        header = frames[0]
        frames = frames[1:]

        n_video_frames = sum(
            1 for f in frames
            if len(f) > 8 and (f[5:8] == iframe_sig or f[5:8] == pframe_sig)
        )
        if end_frame < 0:
            end_frame = n_video_frames

        # Step 3: apply datamoshing effect
        with open(output_avi, 'wb') as out:
            out.write(header)

            if delta > 0:
                # Delta repeat: capture N p-frames then loop them
                if delta > max(1, end_frame - start_frame):
                    return {
                        'success': False,
                        'error': f'Delta ({delta}) is larger than the frame range ({end_frame - start_frame}). '
                                 f'Try a smaller delta or a wider frame range.'
                    }

                repeat_frames = []
                repeat_index = 0

                for index, frame in enumerate(frames):
                    is_video = len(frame) > 8 and (frame[5:8] == iframe_sig or frame[5:8] == pframe_sig)
                    in_range = start_frame <= index < end_frame

                    if not is_video or not in_range:
                        out.write(frame_marker + frame)
                        continue

                    if len(repeat_frames) < delta and frame[5:8] != iframe_sig:
                        repeat_frames.append(frame)
                        out.write(frame_marker + frame)
                    elif len(repeat_frames) == delta:
                        out.write(frame_marker + repeat_frames[repeat_index])
                        repeat_index = (repeat_index + 1) % delta
                    else:
                        out.write(frame_marker + frame)
            else:
                # I-frame removal: drop keyframes so previous frame bleeds through
                for index, frame in enumerate(frames):
                    skip = (
                        len(frame) > 8
                        and frame[5:8] == iframe_sig
                        and start_frame <= index <= end_frame
                    )
                    if not skip:
                        out.write(frame_marker + frame)

        # Step 4: convert output AVI to mp4
        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-i', output_avi,
            '-crf', '18', '-pix_fmt', 'yuv420p',
            '-vcodec', 'libx264', '-acodec', 'aac',
            '-b', '10000k', '-r', str(fps),
            output_path
        ])

        if os.path.exists(output_path):
            return {'success': True}
        return {'success': False, 'error': 'Output file was not created. ffmpeg may have failed.'}

    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        for tmp in [input_avi, output_avi]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


def run_mosh_job(job_id: str, input_path: str, output_path: str, params: dict):
    processing_jobs[job_id] = {
        'status': 'processing',
        'progress': 0,
        'error': None,
        'params': params
    }

    def worker():
        try:
            result = run_mosh(input_path, output_path, params, job_id)
            if result.get('success'):
                processing_jobs[job_id]['status'] = 'completed'
                processing_jobs[job_id]['progress'] = 100
            else:
                processing_jobs[job_id]['status'] = 'failed'
                processing_jobs[job_id]['error'] = result.get('error', 'Unknown error')
                processing_jobs[job_id]['progress'] = 100
        except Exception as e:
            processing_jobs[job_id]['status'] = 'failed'
            processing_jobs[job_id]['error'] = str(e)
            processing_jobs[job_id]['progress'] = 100

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process():
    if 'video' not in request.files or request.files['video'].filename == '':
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    job_id = str(uuid.uuid4())[:8]

    ext = os.path.splitext(video.filename)[1] or '.mp4'
    input_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_input{ext}')
    output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_moshed.mp4')
    video.save(input_path)

    use_manual = request.form.get('use_manual') == 'true'

    if use_manual:
        effect = request.form.get('effect', 'iframe_removal')
        params = {
            'effect': effect,
            'start_frame': int(request.form.get('start_frame', 0)),
            'end_frame': int(request.form.get('end_frame', -1)),
            'fps': int(request.form.get('fps', 30)),
            'delta': int(request.form.get('delta', 5)),
            'explanation': ''
        }
    else:
        prompt = request.form.get('prompt', '').strip()
        api_key = request.form.get('api_key', '').strip()

        if not prompt:
            return jsonify({'error': 'Please enter a description of the effect.'}), 400
        if not api_key:
            return jsonify({'error': 'OpenAI API key is required for AI mode.'}), 400

        video_metadata = get_video_metadata(input_path)
        params = parse_prompt_with_ai(prompt, api_key, video_metadata)
        if 'error' in params:
            return jsonify({'error': f'AI error: {params["error"]}'}), 400

    # Run mosh processing synchronously so client UI has immediate access to output
    result = run_mosh(input_path, output_path, params, job_id)

    if result.get('success'):
        return jsonify({
            'success': True,
            'job_id': job_id,
            'params': params,
            'explanation': params.get('explanation', ''),
            'download_url': f'/api/download/{job_id}'
        })

    return jsonify({'success': False, 'error': result.get('error', 'Processing failed.')}), 500


@app.route('/api/status/<job_id>')
def status(job_id):
    job_id = ''.join(c for c in job_id if c.isalnum() or c == '-')
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    status_info = processing_jobs[job_id].copy()
    output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_moshed.mp4')
    status_info['downloadable'] = os.path.exists(output_path)
    status_info['download_url'] = f'/api/download/{job_id}' if status_info['downloadable'] else None

    return jsonify(status_info)


@app.route('/api/download/<job_id>')
def download(job_id):
    # Sanitize job_id to prevent path traversal
    job_id = ''.join(c for c in job_id if c.isalnum() or c == '-')
    output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_moshed.mp4')
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True, download_name='moshed.mp4')
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/narrative', methods=['POST'])
def narrative():
    """Generate narrative response based on user input and maintain conversation history."""
    data = request.get_json()
    job_id = data.get('job_id', str(uuid.uuid4())[:8])
    user_input = data.get('message', '').strip()
    api_key = data.get('api_key', '').strip()
    use_history = data.get('use_history', True)

    if not user_input:
        return jsonify({'error': 'Please provide a message.'}), 400
    if not api_key:
        return jsonify({'error': 'API key is required.'}), 400

    # Initialize or retrieve conversation history
    if job_id not in conversation_histories or not use_history:
        conversation_histories[job_id] = [
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT}
        ]

    narrative_response, conversation_histories[job_id] = get_narrative_response(
        user_input, conversation_histories[job_id], api_key
    )

    return jsonify({
        'success': True,
        'job_id': job_id,
        'narrative': narrative_response
    })


@app.route('/api/generate-image', methods=['POST'])
def generate_image_endpoint():
    """Generate an image based on the narrative/dream description."""
    data = request.get_json()
    job_id = data.get('job_id', '').strip()
    prompt = data.get('prompt', '').strip()
    api_key = data.get('api_key', '').strip()

    if not prompt:
        return jsonify({'error': 'Please provide a prompt for image generation.'}), 400
    if not api_key:
        return jsonify({'error': 'API key is required.'}), 400

    # Enhance the prompt for better image generation
    enhanced_prompt = f"A surreal, dreamlike, artistic visualization of: {prompt}"
    image_url = generate_image(enhanced_prompt, api_key)

    if "Error" in image_url:
        return jsonify({'error': image_url}), 500

    return jsonify({
        'success': True,
        'image_url': image_url,
        'prompt': prompt
    })


@app.route('/api/clear-history/<job_id>', methods=['POST'])
def clear_history(job_id):
    """Clear conversation history for a job."""
    if job_id in conversation_histories:
        del conversation_histories[job_id]
    return jsonify({'success': True})


@app.route('/api/ping')
def ping():
    ffmpeg_path = ffmpeg_cmd()
    ffmpeg_ok = False
    try:
        if ffmpeg_path == 'ffmpeg':
            # best effort if not found in PATH
            ffmpeg_ok = subprocess.call([ffmpeg_path, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        else:
            ffmpeg_ok = os.path.exists(ffmpeg_path) and subprocess.call([ffmpeg_path, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except FileNotFoundError:
        ffmpeg_ok = False
    except Exception:
        ffmpeg_ok = False

    return jsonify({'ok': True, 'ffmpeg': ffmpeg_ok, 'ffmpeg_path': ffmpeg_path, 'hint': user_ffmpeg_install_hint()})


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'error': 'Uploaded video is too large. Limit is 500 MB.'}), 413


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Resource not found'}), 404


if __name__ == '__main__':
    # Quick startup checks
    print('\n=== Datamosh Studio ===')
    ffmpeg_path = ffmpeg_cmd()
    try:
        ret = subprocess.call([ffmpeg_path, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ret == 0:
            print(f'ffmpeg found: {ffmpeg_path}')
        else:
            print('ffmpeg not found or failed to execute at path:', ffmpeg_path)
            print('  Hint:', user_ffmpeg_install_hint())
    except FileNotFoundError:
        print('ffmpeg not found. Path tried:', ffmpeg_path)
        print('  Hint:', user_ffmpeg_install_hint())
    except Exception as e:
        print('ffmpeg check error:', e)
        print('  Hint:', user_ffmpeg_install_hint())

    host = '0.0.0.0'
    port = 5000
    if is_port_in_use('127.0.0.1', port):
        print(f'Cannot start: port {port} is already in use.')
        print('Close the other server/process, then try again.')
        print('Windows helpers:')
        print('  netstat -ano | findstr :5000')
        print('  Stop-Process -Id <PID> -Force')
        sys.exit(1)

    print(f'Open http://127.0.0.1:{port}')
    app.run(debug=False, host=host, port=port)
