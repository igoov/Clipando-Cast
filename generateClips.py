import os
import subprocess
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import whisper
import requests
import argparse
import time
import textwrap
from typing import List, Dict, Any, Tuple
import google.genai as genai
from google.genai import types
from collections import deque
import shutil
try:
    from scipy.io import wavfile
    from scipy.signal import find_peaks
except ImportError:
    wavfile = None
    find_peaks = None

# ------------ global logger ------------
LOG_PATH = None

def log(msg):
    print(msg)
    global LOG_PATH
    if LOG_PATH:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception:
            pass

# ------------ helper utilities ------------

def parse_timestamp(ts: str) -> float:
    """Converte string MM:SS ou HH:MM:SS em segundos."""
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0.0
        m, s = parts
    else:
        return float(parts[0])
    return h * 3600 + m * 60 + s


def extract_audio(video_path, output_path="temp_audio.wav"):
    """Extract audio from video file using ffmpeg."""
    command = f'ffmpeg -i "{video_path}" -ab 160k -ac 2 -ar 44100 -vn "{output_path}" -y'
    subprocess.call(command, shell=True)
    return output_path


def transcribe_audio(audio_path, whisper_model_size="base"):
    """Transcribe audio using Whisper and return list of segments."""
    log("Loading Whisper model...")
    model = whisper.load_model(whisper_model_size)
    log(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True)
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
    """Prompt the user to review and optionally correct transcription."""
    log("\n=== Transcription Review ===")
    for i, segment in enumerate(transcription_segments):
        log(f"\nSegment {i+1}/{len(transcription_segments)}")
        log(f"[{segment['start']:.2f} - {segment['end']:.2f}]")
        log(f"Current text: {segment['text']}")
        user_input = input("Correction (or enter to accept, q to quit): ")
        if user_input.lower() == 'q':
            break
        elif user_input:
            segment['text'] = user_input
    return transcription_segments


def agrupar_blocos_legendas(word_timings: List[Dict[str, Any]], max_words: int = 4) -> List[Dict[str, Any]]:
    """Agrupa palavras em blocos de 2-4 para legendas mais naturais."""
    articles = {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
                "no", "na", "nos", "nas", "e"}
    blocks = []
    current = []
    for idx, w in enumerate(word_timings):
        current.append(w)
        finalize = False
        if any(p in w.get("text", "") for p in ".?!"):
            finalize = True
        if len(current) >= max_words:
            finalize = True
        if not finalize and idx + 1 < len(word_timings):
            next_word = word_timings[idx + 1]["text"].lower().strip(".,?!")
            if next_word in articles and len(current) == 1:
                finalize = False
        if finalize:
            start = current[0]["start"]
            end = current[-1]["end"]
            text = " ".join([x["text"] for x in current])
            blocks.append({"text": text, "start": start, "end": end})
            current = []
    if current:
        start = current[0]["start"]
        end = current[-1]["end"]
        text = " ".join([x["text"] for x in current])
        blocks.append({"text": text, "start": start, "end": end})
    return blocks


def split_lines(draw, text: str, font: ImageFont.FreeTypeFont, max_width: int, max_lines: int = 2) -> List[str]:
    """Quebra texto em até ``max_lines`` linhas que caibam em ``max_width`` pixels."""
    words = text.split()
    lines = []
    current = []
    for w in words:
        test = " ".join(current + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width and len(lines) < max_lines:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(" ".join(current))
    return lines


def cv2_to_pil(cv2_img):
    """Convert CV2 BGR image to PIL RGB"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    """Convert PIL RGB to CV2 BGR"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_rounded_rectangle(draw, bbox, radius, fill):
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill)
    draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill)
    draw.pieslice((x1, y1, x1 + radius * 2, y1 + radius * 2), 180, 270, fill=fill)
    draw.pieslice((x2 - radius * 2, y1, x2, y1 + radius * 2), 270, 360, fill=fill)
    draw.pieslice((x1, y2 - radius * 2, x1 + radius * 2, y2), 90, 180, fill=fill)
    draw.pieslice((x2 - radius * 2, y2 - radius * 2, x2, y2), 0, 90, fill=fill)


def gerar_legendas_palavra_por_palavra(transcricao: List[Dict[str, Any]]):
    words = []
    for seg in transcricao:
        for w in seg.get("words", []):
            txt = w.get("word", "").strip()
            if txt:
                words.append({"text": txt, "start": w["start"], "end": w["end"]})
    return words


def detectar_idioma_simples(texto: str) -> str:
    """Detecta idioma baseado em palavras comuns."""
    if not texto:
        return "pt"
    
    texto = texto.lower()
    
    # Palavras comuns em cada idioma
    pt_words = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'mais']
    en_words = ['the', 'of', 'and', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'for', 'on']
    es_words = ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del', 'con', 'por']
    
    # Contar ocorrências
    palavras = texto.split()
    count_pt = sum(1 for w in palavras if w in pt_words)
    count_en = sum(1 for w in palavras if w in en_words)
    count_es = sum(1 for w in palavras if w in es_words)
    
    # Determinar idioma
    if count_pt >= count_en and count_pt >= count_es:
        return "pt"
    elif count_en >= count_pt and count_en >= count_es:
        return "en"
    else:
        return "es"


def detectar_risadas(audio_path: str) -> List[Tuple[float, float]]:
    """Detecta risadas no áudio (versão simplificada)."""
    laughter_intervals = []
    
    if wavfile is None or find_peaks is None:
        return laughter_intervals
    
    try:
        rate, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        # Normalizar
        data = data.astype(np.float64)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        
        # Análise de espectro para detectar risadas (faixa de frequência 500-2000 Hz)
        from scipy import signal
        
        # Parâmetros
        frame_duration = 0.5  # 500ms frames
        frame_size = int(rate * frame_duration)
        hop = int(rate * 0.25)  # 250ms hop
        
        laughter_patterns = []
        
        for i in range(0, len(data) - frame_size, hop):
            frame = data[i:i + frame_size]
            
            # Calcular espectro
            freqs, psd = signal.welch(frame, rate, nperseg=min(1024, len(frame)))
            
            # Energia na faixa de risada (500-2000 Hz)
            mask = (freqs >= 500) & (freqs <= 2000)
            laughter_energy = np.sum(psd[mask])
            
            # Energia total
            total_energy = np.sum(psd)
            
            # Proporção de energia na faixa de risada
            if total_energy > 0:
                laughter_ratio = laughter_energy / total_energy
                laughter_patterns.append(laughter_ratio)
            else:
                laughter_patterns.append(0)
        
        # Detectar picos de proporção de risada
        laughter_patterns = np.array(laughter_patterns)
        if len(laughter_patterns) > 0 and np.max(laughter_patterns) > 0.3:  # Threshold empírico
            threshold = np.mean(laughter_patterns) + 1.5 * np.std(laughter_patterns)
            peaks, _ = find_peaks(laughter_patterns, height=threshold, distance=3)
            
            # Agrupar picos próximos
            if len(peaks) > 0:
                in_laughter = False
                start_idx = 0
                
                for i, is_peak in enumerate([i in peaks for i in range(len(laughter_patterns))]):
                    if is_peak and not in_laughter:
                        in_laughter = True
                        start_idx = i
                    elif not is_peak and in_laughter:
                        in_laughter = False
                        if i - start_idx >= 2:  # Mínimo 1 segundo (2 * 0.5s)
                            start_time = start_idx * hop / rate
                            end_time = i * hop / rate + frame_duration
                            laughter_intervals.append((start_time, end_time))
                
                # Último segmento
                if in_laughter:
                    start_time = start_idx * hop / rate
                    end_time = len(laughter_patterns) * hop / rate + frame_duration
                    laughter_intervals.append((start_time, end_time))
        
    except Exception as e:
        log(f"[debug] Erro na detecção de risadas: {e}")
    
    return laughter_intervals


def aplicar_fade_audio(video_path: str, fade_in_duration: float = 1.0, fade_out_duration: float = 1.0) -> str:
    """
    Aplica fade in e fade out no áudio do vídeo.
    Retorna o caminho do novo vídeo com fade aplicado.
    """
    if not os.path.exists(video_path):
        return video_path
    
    output_path = video_path.replace('.mp4', '_fade.mp4')
    
    # Obter duração do vídeo
    try:
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        duracao = float(subprocess.check_output(cmd, shell=True).decode().strip())
    except:
        duracao = 10  # fallback
    
    # Aplicar fade in e fade out com ffmpeg
    fade_cmd = [
        'ffmpeg', '-i', video_path,
        '-af', f'afade=t=in:ss=0:d={fade_in_duration},afade=t=out:st={duracao - fade_out_duration}:d={fade_out_duration}',
        '-c:v', 'copy',
        '-y', output_path
    ]
    
    try:
        subprocess.run(' '.join(fade_cmd), shell=True, check=True, capture_output=True)
        log(f"Fade in/out aplicado ao áudio: {fade_in_duration}s in, {fade_out_duration}s out")
        
        # Substituir vídeo original
        os.remove(video_path)
        os.rename(output_path, video_path)
    except Exception as e:
        log(f"[erro] Falha ao aplicar fade no áudio: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
    
    return video_path


def detectar_momentos_virais(audio_path: str, 
                            transcription_segments: List[Dict[str, Any]] = None,
                            video_path: str = None,
                            threshold_volume: float = 0.7,
                            threshold_energy_change: float = 2.0) -> List[Dict[str, Any]]:
    """
    Detecta momentos potencialmente virais usando múltiplas estratégias:
    - Volume alto
    - Mudanças bruscas de energia (explosões, risadas)
    - Palavras-chave em múltiplos idiomas
    - Padrões de edição (cortes rápidos) - opcional
    - Detecção de risadas/apupos
    """
    intervals = []
    result = []
    
    # 1. DETECÇÃO POR ÁUDIO AVANÇADA
    if wavfile is not None:
        try:
            rate, data = wavfile.read(audio_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            
            # Normalizar áudio
            data = data.astype(np.float64)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
            
            frame_size = int(rate * 0.2)  # 200ms frames
            hop = int(rate * 0.05)  # 50ms hop
            
            energies = []
            timestamps = []
            
            for i in range(0, len(data) - frame_size, hop):
                frame = data[i:i + frame_size]
                energy = np.sum(frame ** 2)
                energies.append(energy)
                timestamps.append(i / rate)
            
            energies = np.array(energies)
            
            # Detectar picos de volume
            if len(energies) > 0 and np.max(energies) > 0:
                # Usar percentil dinâmico em vez de threshold fixo
                dynamic_threshold = np.percentile(energies, 90)  # Top 10%
                
                # Detectar mudanças bruscas de energia (explosões)
                if len(energies) > 1:
                    energy_diff = np.diff(energies)
                    energy_diff = np.append(0, energy_diff)  # manter mesmo tamanho
                    
                    # Detectar momentos de alta energia
                    high_energy = energies > dynamic_threshold
                    
                    # Detectar explosões (aumento repentino)
                    explosions = energy_diff > threshold_energy_change * np.mean(np.abs(energy_diff))
                    
                    # Combinar detecções
                    viral_audio = high_energy | explosions
                    
                    # Agrupar frames consecutivos
                    in_viral = False
                    start_time = 0
                    
                    for i, is_viral in enumerate(viral_audio):
                        if is_viral and not in_viral:
                            in_viral = True
                            start_time = timestamps[i]
                        elif not is_viral and in_viral:
                            in_viral = False
                            end_time = timestamps[i]
                            if end_time - start_time >= 2:  # mínimo 2 segundos
                                intervals.append((start_time, end_time, "audio_peak"))
                        elif in_viral and i == len(viral_audio) - 1:
                            end_time = timestamps[i] + hop/rate
                            if end_time - start_time >= 2:
                                intervals.append((start_time, end_time, "audio_peak"))
                            
        except Exception as e:
            log(f"[debug] Detecção de áudio falhou: {e}")
    else:
        log("[warning] scipy not installed; audio-based viral detection disabled")
    
    # Converter intervalos de áudio para resultado
    for s, e, src in intervals:
        # Expandir um pouco para capturar contexto
        s_ext = max(0, s - 1.0)
        e_ext = min(e + 2.0, s_ext + 20)
        result.append({
            "start": s_ext, 
            "end": e_ext, 
            "score": 1.5, 
            "source": src
        })
    
    # 2. DETECÇÃO POR PALAVRAS-CHAVE (MULTI-IDIOMA)
    if transcription_segments:
        # Dicionário de palavras-chave por idioma e peso
        keywords_db = {
            "pt": {
                "alto impacto": ["segredo", "revelação", "chocante", "incrível", "surpreendente",
                               "nunca", "não vai acreditar", "você não vai acreditar", "descobri", "putz",
                               "caramba", "nossa", "sério", "mentira", "impossível", "absurdo",
                               "bizarro", "assustador", "perigoso", "urgente", "atenção"],
                "engajamento": ["inscreva-se", "comenta", "compartilha", "like", "curte",
                               "deixa o like", "ativa o sininho", "segue", "compartilhe"],
                "emocional": ["chorei", "ri", "rindo", "emocionante", "triste", "feliz", "raiva",
                             "ódio", "amor", "paixão", "medo", "ansiedade", "alívio"],
                "perguntas": ["como", "quando", "onde", "por que", "porquê", "qual", "quem",
                             "você sabia", "você já", "já pensou"]
            },
            "en": {
                "alto impacto": ["secret", "reveal", "shocking", "incredible", "surprising",
                               "never", "you won't believe", "discovered", "wow", "oh my god",
                               "seriously", "no way", "impossible", "crazy", "insane",
                               "scary", "dangerous", "urgent", "attention"],
                "engajamento": ["subscribe", "comment", "share", "like", "hit that like",
                               "ring the bell", "follow"],
                "emocional": ["cried", "laughed", "laughing", "emotional", "sad", "happy", "angry",
                             "hate", "love", "passion", "fear", "anxiety", "relief"],
                "perguntas": ["how", "when", "where", "why", "what", "who", "did you know",
                             "have you", "ever wondered"]
            },
            "es": {
                "alto impacto": ["secreto", "revelación", "impactante", "increíble", "sorprendente",
                               "nunca", "no vas a creer", "descubrí", "guau", "dios mío",
                               "en serio", "no puede ser", "imposible", "loco", "insano",
                               "aterrador", "peligroso", "urgente", "atención"],
                "engajamento": ["suscríbete", "comenta", "comparte", "like", "dale like",
                               "activa la campanita", "sigue"],
                "emocional": ["lloré", "reí", "riendo", "emocionante", "triste", "feliz", "enojo",
                             "odio", "amor", "pasión", "miedo", "ansiedad", "alivio"],
                "perguntas": ["cómo", "cuándo", "dónde", "por qué", "cuál", "quién",
                             "sabías que", "has", "alguna vez"]
            }
        }
        
        # Detectar idioma
        textos = " ".join([seg.get("text", "") for seg in transcription_segments[:10]])
        idioma = detectar_idioma_simples(textos)
        
        keywords_atual = keywords_db.get(idioma, keywords_db["pt"])
        
        for seg in transcription_segments:
            text = seg.get("text", "").lower()
            start_time = seg.get("start", 0)
            end_time = seg.get("end", start_time + 5)
            
            max_score = 0
            matched_keyword = None
            matched_category = None
            
            for categoria, palavras in keywords_atual.items():
                for kw in palavras:
                    if kw in text:
                        # Peso diferente por categoria
                        if categoria == "alto impacto":
                            score = 2.0
                        elif categoria == "engajamento":
                            score = 1.5
                        elif categoria == "perguntas":
                            score = 1.3
                        else:  # emocional
                            score = 1.8
                            
                        # Bônus para palavras-chave exatas vs parciais
                        if kw == text.strip():
                            score *= 1.5
                            
                        if score > max_score:
                            max_score = score
                            matched_keyword = kw
                            matched_category = categoria
            
            if max_score > 0:
                # Verificar se já existe um intervalo próximo
                overlapping = False
                for r in result:
                    if abs(r["start"] - start_time) < 3:
                        # Combinar scores se for próximo
                        r["score"] = max(r["score"], max_score)
                        if r["source"] != "keyword":
                            r["source"] = f"{r['source']}+keyword"
                        overlapping = True
                        break
                
                if not overlapping:
                    result.append({
                        "start": max(0, start_time - 1.5),
                        "end": min(end_time + 3, start_time + 15),
                        "score": max_score,
                        "source": "keyword",
                        "keyword": matched_keyword,
                        "category": matched_category
                    })
    
    # 3. DETECÇÃO POR RISADAS/APLAUSOS
    if wavfile is not None:
        try:
            # Detectar risadas
            laughter_intervals = detectar_risadas(audio_path)
            for s, e in laughter_intervals:
                result.append({
                    "start": s,
                    "end": e,
                    "score": 2.5,  # Risada é muito viral
                    "source": "laughter"
                })
        except Exception as e:
            log(f"[debug] Erro na detecção de risadas: {e}")
    
    # 4. ORDENAR POR SCORE E REMOVER DUPLICATAS SOBREPOSTAS
    if result:
        # Ordenar por score (maior primeiro)
        result.sort(key=lambda x: x["score"], reverse=True)
        
        # Remover intervalos muito sobrepostos (manter o de maior score)
        final_result = []
        for r in result:
            overlap = False
            for fr in final_result:
                # Calcular sobreposição
                overlap_start = max(r["start"], fr["start"])
                overlap_end = min(r["end"], fr["end"])
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    r_duration = r["end"] - r["start"]
                    if overlap_duration > 0.5 * r_duration:
                        overlap = True
                        break
            if not overlap:
                final_result.append(r)
        
        # Limitar a no máximo 15 momentos virais
        final_result = final_result[:15]
        
        return final_result
    
    return result


def rankear_clipes_por_ia(clipes_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Função que rankeia os clipes usando IA (Gemini) ou lógica avançada.
    Retorna a lista ordenada com notas de 0-10.
    """
    log("\n=== RANKEANDO CLIPES POR IA ===")
    
    if not clipes_info:
        return []
    
    # Tentar usar Gemini se disponível
    api_key = os.getenv("GEMINI_API_KEY")
    use_gemini = bool(api_key)
    
    if use_gemini:
        try:
            client = genai.Client(api_key=api_key)
            
            # Preparar prompt para o Gemini
            prompt = "Analise os seguintes clipes de vídeo e dê uma nota de 0 a 10 para cada um, considerando:\n"
            prompt += "- Potencial viral (0-10)\n"
            prompt += "- Engajamento esperado (0-10)\n"
            prompt += "- Qualidade do conteúdo (0-10)\n"
            prompt += "- Clareza da mensagem (0-10)\n"
            prompt += "- Emoção transmitida (0-10)\n\n"
            prompt += "Formato de resposta: JSON com array de objetos contendo 'clip_id', 'nota_final', 'justificativa'\n\n"
            prompt += "Clipes:\n"
            
            for i, clip in enumerate(clipes_info):
                prompt += f"\nClip {i+1}:\n"
                prompt += f"Texto: {clip.get('texto', '')[:200]}...\n"
                prompt += f"Duração: {clip.get('duracao', 0):.1f}s\n"
                prompt += f"Palavras-chave: {', '.join(clip.get('palavras_chave', []))}\n"
                prompt += f"Momentos virais: {clip.get('viral_score', 0)}\n"
            
            prompt += "\nResponda apenas com o JSON, sem texto adicional."
            
            # Chamar Gemini
            response = client.models.generate_content(
                model="gemini-pro",
                contents=prompt
            )
            
            # Tentar parsear resposta
            try:
                # Extrair JSON da resposta
                resposta_texto = response.text
                # Encontrar JSON na resposta
                json_match = re.search(r'\[.*\]', resposta_texto, re.DOTALL)
                if json_match:
                    ranking_data = json.loads(json_match.group())
                    
                    # Aplicar notas aos clipes
                    for item in ranking_data:
                        clip_id = item.get('clip_id', 0)
                        if 0 <= clip_id - 1 < len(clipes_info):
                            clipes_info[clip_id - 1]['nota_ia'] = item.get('nota_final', 5)
                            clipes_info[clip_id - 1]['justificativa_ia'] = item.get('justificativa', '')
                    
                    log("Ranking gerado com sucesso pelo Gemini!")
                else:
                    log("[aviso] Não foi possível extrair JSON da resposta do Gemini")
                    use_gemini = False
            except Exception as e:
                log(f"[aviso] Erro ao processar resposta do Gemini: {e}")
                use_gemini = False
                
        except Exception as e:
            log(f"[aviso] Erro ao usar Gemini para ranking: {e}")
            use_gemini = False
    
    # Fallback: se Gemini não funcionar, usar lógica avançada
    if not use_gemini:
        log("Usando lógica avançada para ranking (fallback)")
        
        for clip in clipes_info:
            nota = 5.0  # Nota base
            
            # Fatores positivos
            if clip.get('viral_score', 0) > 2.0:
                nota += 2.0
            elif clip.get('viral_score', 0) > 1.5:
                nota += 1.0
            elif clip.get('viral_score', 0) > 1.0:
                nota += 0.5
            
            # Palavras-chave de alto impacto
            palavras_impacto = clip.get('palavras_chave', [])
            for p in palavras_impacto:
                if p in ['segredo', 'revelação', 'chocante', 'incrível', 'urgente']:
                    nota += 0.5
                    break
            
            # Duração ideal (15-30s é melhor)
            duracao = clip.get('duracao', 0)
            if 15 <= duracao <= 30:
                nota += 1.0
            elif duracao < 10:
                nota -= 1.0
            elif duracao > 45:
                nota -= 0.5
            
            # Posição no vídeo (primeiros minutos são melhores)
            posicao = clip.get('start_time', 0)
            if posicao < 60:  # Primeiro minuto
                nota += 1.0
            elif posicao < 180:  # Até 3 minutos
                nota += 0.5
            elif posicao > 600:  # Depois de 10 minutos
                nota -= 0.5
            
            # Riso detectado
            if clip.get('tem_risada', False):
                nota += 1.5
            
            # Pergunta (engaja comentários)
            if clip.get('tem_pergunta', False):
                nota += 0.8
            
            # Normalizar para 0-10
            nota = max(0, min(10, nota))
            clip['nota_ia'] = round(nota, 1)
            clip['justificativa_ia'] = f"Nota calculada: viral={clip.get('viral_score',0)}, duração={duracao:.0f}s, posição={posicao:.0f}s"
    
    # Ordenar por nota (maior para menor)
    clipes_ordenados = sorted(clipes_info, key=lambda x: x.get('nota_ia', 0), reverse=True)
    
    # Adicionar posição no ranking
    for i, clip in enumerate(clipes_ordenados):
        clip['ranking'] = i + 1
        clip['estrelas'] = '★' * int(clip.get('nota_ia', 0) / 2) + '☆' * (5 - int(clip.get('nota_ia', 0) / 2))
        log(f"Clip {clip['ranking']}: Nota {clip['nota_ia']}/10 {clip['estrelas']}")
    
    return clipes_ordenados


def adicionar_ranking_ao_video(video_path: str, ranking_info: Dict[str, Any], output_path: str = None) -> str:
    """
    Adiciona um overlay com informações de ranking no final do vídeo.
    """
    if not os.path.exists(video_path):
        log(f"[erro] Vídeo não encontrado: {video_path}")
        return video_path
    
    if output_path is None:
        output_path = video_path.replace('.mp4', '_com_ranking.mp4')
    
    # Obter duração do vídeo
    try:
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        duracao = float(subprocess.check_output(cmd, shell=True).decode().strip())
    except:
        duracao = 10  # fallback
    
    # Criar arquivo de legenda com ranking
    srt_path = video_path.replace('.mp4', '_ranking.srt')
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        # Overlay no final do vídeo (últimos 3 segundos)
        inicio_overlay = max(0, duracao - 3)
        
        # Texto do ranking
        texto_ranking = f"NOTA: {ranking_info.get('nota_ia', '?')}/10\n"
        texto_ranking += f"{ranking_info.get('estrelas', '')}\n"
        texto_ranking += f"Ranking: #{ranking_info.get('ranking', '?')}"
        
        f.write("1\n")
        f.write(f"{formatar_timestamp_srt(inicio_overlay)} --> {formatar_timestamp_srt(duracao)}\n")
        f.write(f"{texto_ranking}\n\n")
        
        # Se houver justificativa, adicionar
        if ranking_info.get('justificativa_ia'):
            f.write("2\n")
            f.write(f"{formatar_timestamp_srt(inicio_overlay + 0.1)} --> {formatar_timestamp_srt(duracao)}\n")
            f.write(f"{ranking_info['justificativa_ia']}\n")
    
    # Aplicar legenda ao vídeo
    output_temp = video_path.replace('.mp4', '_temp_ranking.mp4')
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=3,Outline=1'",
        '-c:a', 'copy',
        '-y', output_temp
    ]
    
    try:
        subprocess.run(' '.join(cmd), shell=True, check=True, capture_output=True)
        
        # Substituir vídeo original se não for especificado output_path diferente
        if output_path == video_path:
            os.remove(video_path)
            os.rename(output_temp, video_path)
        else:
            os.rename(output_temp, output_path)
        
        log(f"Ranking adicionado ao vídeo: {output_path}")
    except Exception as e:
        log(f"[erro] Falha ao adicionar ranking: {e}")
        return video_path
    finally:
        # Limpar arquivo temporário
        if os.path.exists(srt_path):
            os.remove(srt_path)
    
    return output_path


def formatar_timestamp_srt(segundos: float) -> str:
    """Formata timestamp para formato SRT (HH:MM:SS,mmm)."""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    milissegundos = int((segundos - int(segundos)) * 1000)
    return f"{horas:02d}:{minutos:02d}:{segs:02d},{milissegundos:03d}"


# face cascade
autoPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(autoPath)

class LLMClipFinder:
    def __init__(self, api_key=None, model="gemini-pro"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.use_gemini = bool(self.api_key)
        if self.use_gemini:
            self.client = genai.Client(api_key=self.api_key)
        else:
            log("No Google Gemini API key found. Falling back to simple selector.")

    def find_interesting_moments(self, segments: List[Dict[str,Any]], min_clips=3, max_clips=8):
        if self.use_gemini:
            try:
                text = "\n".join([seg.get("text","") for seg in segments])
                prompt = f"Given the following transcript, suggest {min_clips}-{max_clips}" \
                         " interesting clip intervals in MM:SS format with reasons.\n" + text
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                content = response.text
                # attempt to parse JSON from reply
                try:
                    data = json.loads(content)
                    return data
                except Exception:
                    log("[warning] could not parse LLM response, falling back")
            except Exception as e:
                log(f"[warning] LLM call failed: {e}")
        # fallback simple spacing
        clips = []
        total_dur = 0
        if segments:
            total_dur = segments[-1].get("end", 0)
        for i in range(min_clips):
            start = min(i * 30, max(0, total_dur - 15))
            end = min(start + 30, total_dur)
            clips.append({
                "start": f"{int(start//60):02d}:{int(start%60):02d}",
                "end": f"{int(end//60):02d}:{int(end%60):02d}",
                "reason": "fallback"
            })
        return {"clips": clips}


def create_clip(video_path, clip, output_path, captions=True, bg_color=(255, 255, 255, 230),
                highlight_color=(255, 226, 165, 220), text_color=(0, 0, 0), headline_text=None,
                broll_map=None, remove_silence=True, retranscribe=False, ranking_info=None,
                fade_duration=1.0):
    """Render a single clip with enhanced visuals and clean captions."""
    start_time = parse_timestamp(clip["start"])
    end_time = parse_timestamp(clip["end"])
    duration = end_time - start_time
    temp_video = f"{output_path}_temp.mp4"
    extract_cmd = f'ffmpeg -ss {start_time} -i "{video_path}" -t {duration} -c:v copy -c:a copy "{temp_video}" -y'
    log(f"Extracting clip: {extract_cmd}")
    subprocess.call(extract_cmd, shell=True)
    
    if remove_silence:
        cleaned = f"{output_path}_ns.mp4"
        silence_cmd = (
            f'ffmpeg -i "{temp_video}" -af "silenceremove=start_periods=1:'
            f'start_silence=0.5:start_threshold=-50dB" -c:v copy -c:a aac "{cleaned}" -y'
        )
        log(f"Removing silence: {silence_cmd}")
        subprocess.call(silence_cmd, shell=True)
        if os.path.exists(cleaned):
            os.remove(temp_video)
            temp_video = cleaned
            if retranscribe:
                log("Re-transcribing cleaned clip for accurate captions...")
                fresh_segments = transcribe_audio(temp_video)
                clip["segments"] = fresh_segments
                duration = float(subprocess.check_output(
                    f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{temp_video}"',
                    shell=True
                ).decode().strip())
            else:
                log("Skipping re-transcription (retranscribe=False)")
    
    cap = cv2.VideoCapture(temp_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_width = 1080
    target_height = 1920
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{output_path}_processed.mp4", fourcc, fps, (target_width, target_height))
    
    if not out.isOpened():
        log("Failed to open video writer.")
        return None
    
    # prepare word and phrase timings
    word_timings = []
    if clip.get("segments"):
        raw_words = gerar_legendas_palavra_por_palavra(clip["segments"])
        for w in raw_words:
            wstart = max(0, w["start"] - start_time)
            wend = min(duration, w["end"] - start_time)
            word_timings.append({"text": w["text"], "start": wstart, "end": wend})
    
    if not word_timings and clip.get("segments"):
        for seg in clip["segments"]:
            text = seg.get("text", "").strip()
            if text:
                word_timings.append({"text": text, "start": 0, "end": duration})
    
    if not word_timings:
        fallback_text = clip.get("caption") or "[sem legenda]"
        word_timings.append({"text": fallback_text, "start": 0, "end": duration})
        log(f"[debug] inserted fallback caption '{fallback_text}' for clip at {start_time}")
    
    phrase_timings = agrupar_blocos_legendas(word_timings)
    log(f"[debug] word_timings count={len(word_timings)}, phrase_timings count={len(phrase_timings)} for clip starting at {start_time}")
    
    if captions and not phrase_timings:
        log(f"[warning] captions requested but no phrase timings for clip at {start_time}")
    
    emphasis_intervals = []
    for w in word_timings:
        if any(p in w["text"] for p in ['!', '?']) or len(w["text"]) > 8:
            emphasis_intervals.append((max(0, w["start"] - 0.2), min(duration, w["end"] + 0.2)))
    
    # prepare b-roll intervals map from keywords
    broll_intervals = []
    if broll_map:
        for w in word_timings:
            key = w["text"].lower().strip('.,?!')
            if key in broll_map:
                broll_intervals.append({
                    "start": w["start"],
                    "end": min(duration, w["end"] + 1.0),
                    "image": broll_map[key]
                })
    
    # ============================================
    # CSRT TRACKER COM SUAVIZAÇÃO EXTREMA
    # ============================================
    
    # CSRT Tracker (mais preciso para rostos)
    tracker = None
    tracker_inicializado = False
    face_positions = deque(maxlen=150)
    missed_face = 0
    
    # Variáveis para suavização
    pos_x = orig_width / 2
    pos_y = orig_height / 2
    size_h = orig_height / 3
    alpha = 0.97  # Suavização EXTREMA (0.97 = muito suave)
    ultimo_x_valido = pos_x
    ultimo_y_valido = pos_y
    ultimo_h_valido = size_h
    
    log(f"Processing {total_frames} frames at {fps} fps (duration {duration:.1f}s)")
    frames_processed = 0
    
    while frames_processed < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frames_processed / fps
        frames_processed += 1
        
        if frames_processed % 100 == 0:
            log(f"Progress: {frames_processed}/{total_frames} frames ({frames_processed/total_frames*100:.1f}%)")
        
        zoom = 1.0
        for es, ee in emphasis_intervals:
            if es <= current_time <= ee:
                rel = (current_time - es) / max(ee - es, 0.001)
                zoom = 1.0 + 0.05 * (1 - abs(2 * rel - 1))
                break
        
        # ============================================
        # CSRT TRACKING
        # ============================================
        
        # Inicializar tracker no primeiro frame ou se perdeu
        if not tracker_inicializado or missed_face > 30:
            # Detectar face com Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(80, 80)
            )
            
            if len(faces) > 0:
                # Pegar o maior rosto
                x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
                
                # Inicializar tracker CSRT
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                tracker_inicializado = True
                missed_face = 0
                
                # Posição inicial
                cx = x + w/2
                cy = y + h/2
                ultimo_x_valido = cx
                ultimo_y_valido = cy
                ultimo_h_valido = h
                
                if frames_processed % 500 == 0:
                    log(f"[debug] CSRT tracker inicializado em ({cx:.1f},{cy:.1f})")
            else:
                # Sem face detectada
                cx = ultimo_x_valido
                cy = ultimo_y_valido
                h = ultimo_h_valido
        
        # Usar tracker se inicializado
        if tracker_inicializado and tracker is not None:
            success, bbox = tracker.update(frame)
            
            if success:
                missed_face = 0
                x, y, w, h = [int(v) for v in bbox]
                
                # Verificar se o tracker não encolheu demais
                if w > 30 and h > 30:
                    cx = x + w/2
                    cy = y + h/2
                    
                    # TRAVA DE POSIÇÃO - limitar movimento brusco
                    dx = abs(cx - ultimo_x_valido)
                    dy = abs(cy - ultimo_y_valido)
                    
                    if dx > 40 or dy > 40:
                        # Movimento muito brusco! Usar posição anterior
                        if frames_processed % 100 == 0:
                            log(f"[debug] Trava ativada! Movimento: ({dx:.1f},{dy:.1f})")
                        cx = ultimo_x_valido
                        cy = ultimo_y_valido
                        h = ultimo_h_valido
                    else:
                        # Movimento normal, atualizar
                        ultimo_x_valido = cx
                        ultimo_y_valido = cy
                        ultimo_h_valido = h
                else:
                    # Tracker encolheu, usar posição anterior
                    cx = ultimo_x_valido
                    cy = ultimo_y_valido
                    h = ultimo_h_valido
            else:
                # Tracker perdeu o rosto
                missed_face += 1
                cx = ultimo_x_valido
                cy = ultimo_y_valido
                h = ultimo_h_valido
                
                if missed_face > 30:
                    tracker_inicializado = False
                    if frames_processed % 500 == 0:
                        log(f"[debug] Tracker perdeu rosto, reinicalizando...")
        else:
            # Tracker não inicializado, usar última posição conhecida
            cx = ultimo_x_valido
            cy = ultimo_y_valido
            h = ultimo_h_valido
        
        # ============================================
        # SUAVIZAÇÃO EXPONENCIAL FINAL
        # ============================================
        
        # Aplicar suavização exponencial (ultra suave)
        pos_x = pos_x * alpha + cx * (1 - alpha)
        pos_y = pos_y * alpha + cy * (1 - alpha)
        size_h = size_h * alpha + h * (1 - alpha)
        
        # Guardar posição suavizada
        face_positions.append((pos_x, pos_y, size_h))
        
        # Usar posição suavizada
        avg_cx = pos_x
        avg_cy = pos_y
        avg_h = size_h
        
        # ============================================
        # COMPOSIÇÃO DO FRAME
        # ============================================
        
        # compose frame: blurred background and optional face-centering
        scale_bg = max(target_width/orig_width, target_height/orig_height)
        new_w = int(orig_width * scale_bg)
        new_h = int(orig_height * scale_bg)
        resized_bg = cv2.resize(frame, (new_w, new_h))
        x_off_bg = (new_w - target_width) // 2
        y_off_bg = (new_h - target_height) // 2
        background = resized_bg[y_off_bg:y_off_bg+target_height, x_off_bg:x_off_bg+target_width]
        background = cv2.GaussianBlur(background, (51, 51), 0)
        result = background.copy()

        # overlay b-roll image if available
        overlay_image = None
        for interval in broll_intervals:
            if interval["start"] <= current_time <= interval["end"]:
                overlay_image = interval["image"]
                break
        
        if overlay_image and os.path.exists(overlay_image):
            try:
                img = cv2.imread(overlay_image)
                img = cv2.resize(img, (target_width, target_height))
                result = img
            except Exception:
                pass
        else:
            # compute face-aware crop com posição suavizada
            scale_vid = max(target_width/orig_width, target_height/orig_height) * zoom
            target_ratio = target_width / target_height
            orig_ratio = orig_width / orig_height
            
            if orig_ratio > target_ratio:
                crop_h = orig_height
                crop_w = int(crop_h * target_ratio)
            else:
                crop_w = orig_width
                crop_h = int(crop_w / target_ratio)

            offset_y = int(crop_h * 0.2)
            cx = int(avg_cx)
            cy = int(avg_cy) - offset_y
            
            # Limitar para não sair do frame
            x1 = max(0, min(orig_width - crop_w, cx - crop_w // 2))
            y1 = max(0, min(orig_height - crop_h, cy - crop_h // 2))
            
            cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            resized = cv2.resize(cropped, (target_width, target_height))
            result = resized

        # convert to PIL for drawn elements
        pil_img = cv2_to_pil(result)
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # headline text
        if headline_text:
            try:
                hfont = ImageFont.truetype("C:/Windows/Fonts/ARIALBD.TTF", 120)
            except:
                try:
                    hfont = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 120)
                except:
                    hfont = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), headline_text, font=hfont)
            hw = bbox[2] - bbox[0]
            hh = bbox[3] - bbox[1]
            hx = (target_width - hw) // 2
            hy = 40
            pad = 20
            draw_rounded_rectangle(draw, (hx-pad, hy-pad, hx+hw+pad, hy+hh+pad), 30, (0, 0, 0, 150))
            draw.text((hx, hy), headline_text, font=hfont, fill=(255, 255, 0, 255))

        # dynamic phrase captions
        if phrase_timings:
            if not captions:
                log("[debug] captions flag not set but phrase timings exist -> rendering anyway")
            
            active_phrase = None
            for ph in phrase_timings:
                if ph["start"] <= current_time <= ph["end"]:
                    active_phrase = ph["text"].strip()
                    break
            
            if active_phrase is None and phrase_timings:
                active_phrase = phrase_timings[0]["text"].strip()
            
            if active_phrase:
                base_size = max(90, int(target_width * 0.09))
                if len(active_phrase) > 20:
                    base_size = max(60, base_size - (len(active_phrase)-20))
                
                try:
                    tfont = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", base_size)
                except:
                    try:
                        tfont = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", base_size)
                    except:
                        tfont = ImageFont.load_default()
                
                lines = split_lines(draw, active_phrase, tfont, target_width - 200, max_lines=2)
                ty = target_height - 80
                for line in reversed(lines):
                    bbox = draw.textbbox((0,0), line, font=tfont)
                    txt_w = bbox[2] - bbox[0]
                    txt_h = bbox[3] - bbox[1]
                    tx = (target_width - txt_w) // 2
                    draw.text((tx, ty - txt_h), line, font=tfont, fill="white", stroke_width=4, stroke_fill="black")
                    ty -= (txt_h + 10)

        result = pil_to_cv2(pil_img)
        out.write(result)
    
    cap.release()
    out.release()
    
    final_output = output_path
    if not final_output.lower().endswith('.mp4'):
        final_output += '.mp4'
    
    combine_cmd = f'ffmpeg -i "{output_path}_processed.mp4" -i "{temp_video}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{final_output}" -y'
    log(f"Adding audio: {combine_cmd}")
    subprocess.call(combine_cmd, shell=True)
    
    if os.path.exists(temp_video):
        os.remove(temp_video)
    if os.path.exists(f"{output_path}_processed.mp4"):
        os.remove(f"{output_path}_processed.mp4")
    
    # Aplicar FADE IN e FADE OUT no áudio
    if fade_duration > 0:
        log(f"Aplicando fade in/out no áudio: {fade_duration}s")
        final_output = aplicar_fade_audio(final_output, fade_duration, fade_duration)
    
    # Adicionar ranking ao vídeo se fornecido
    if ranking_info:
        final_com_ranking = final_output.replace('.mp4', '_classificado.mp4')
        adicionar_ranking_ao_video(final_output, ranking_info, final_com_ranking)
        
        # Substituir pelo vídeo com ranking
        if os.path.exists(final_com_ranking):
            os.remove(final_output)
            os.rename(final_com_ranking, final_output)
    
    if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
        log(f"Successfully created clip at {final_output}")
        return final_output
    
    return None


def criar_estrutura_clipes(videos_gerados: List[str], textos: List[str], base_dir: str = None, rankings: List[Dict] = None):
    if base_dir:
        clipes_root = os.path.join(base_dir, "clipes")
    else:
        clipes_root = "clipes"
    
    os.makedirs(clipes_root, exist_ok=True)
    
    for idx, (vid, txt) in enumerate(zip(videos_gerados, textos), start=1):
        # Encontrar ranking para este clipe
        ranking_info = {}
        if rankings and idx - 1 < len(rankings):
            ranking_info = rankings[idx - 1]
        
        title, tags = sugerir_titulo_e_hashtags(txt)
        
        # Adicionar nota ao título se disponível
        if ranking_info and 'nota_ia' in ranking_info:
            estrelas = '★' * int(ranking_info['nota_ia'] / 2) + '☆' * (5 - int(ranking_info['nota_ia'] / 2))
            title = f"[{ranking_info['nota_ia']}/10 {estrelas}] {title}"
        
        safe = re.sub(r"[\\/*?:\"<>|]", "", title).strip()
        if not safe:
            safe = f"clip_{idx}"
        
        folder = os.path.join(clipes_root, f"clip_{idx}_{safe}")
        os.makedirs(folder, exist_ok=True)
        
        dest = os.path.join(folder, os.path.basename(vid))
        try:
            shutil.move(vid, dest)
        except Exception:
            shutil.copy(vid, dest)
        
        with open(os.path.join(folder, "titulo.txt"), "w", encoding="utf-8") as f:
            f.write(title)
        
        with open(os.path.join(folder, "hashtags.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(tags))
        
        # Salvar ranking se disponível
        if ranking_info:
            with open(os.path.join(folder, "ranking.json"), "w", encoding="utf-8") as f:
                json.dump(ranking_info, f, indent=2, ensure_ascii=False)


def sugerir_titulo_e_hashtags(texto_clipe: str) -> Tuple[str, List[str]]:
    title = texto_clipe.strip().split(".")[0][:100]
    hashtags = ["#humor", "#podcast", "#entretenimento"]
    words = [w.strip("#.,?!") for w in texto_clipe.split()]
    for w in words:
        lw = w.lower()
        if len(lw) > 4 and lw not in ("humor", "podcast", "entretenimento"):
            hashtags.append("#" + lw)
        if len(hashtags) >= 10:
            break
    return title, hashtags


def main():
    parser = argparse.ArgumentParser(description="Create video clips using AI to find interesting moments")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-dir", default="ai_clips", help="Directory to save output clips")
    parser.add_argument("--min-clips", type=int, default=3, help="Minimum number of clips to suggest")
    parser.add_argument("--max-clips", type=int, default=8, help="Maximum number of clips to suggest")
    parser.add_argument("--whisper-model", default="base", choices=["tiny","base","small","medium","large"],
                        help="Whisper model size to use for transcription")
    parser.add_argument("--api-key", help="API key for LLM service (optional)")
    parser.add_argument("--captions", dest="captions", action="store_true", help="render captions (default)")
    parser.add_argument("--no-captions", dest="captions", action="store_false", help="disable captions")
    parser.set_defaults(captions=True)
    parser.add_argument("--no-review", action="store_true", help="Skip clip review")
    parser.add_argument("--bg-color", default="255,255,255,230")
    parser.add_argument("--highlight-color", default="255,226,165,220")
    parser.add_argument("--text-color", default="0,0,0")
    parser.add_argument("--headline", default=None)
    parser.add_argument("--retranscribe", action="store_true", help="Re-run transcription on cleaned clip to improve captions")
    parser.add_argument("--broll-dir", help="Directory containing keyword images for b-roll overlays")
    parser.add_argument("--rank-clips", action="store_true", help="Rank clips using AI (nota de 0-10)")
    parser.add_argument("--no-ranking-overlay", action="store_true", help="Don't add ranking overlay to videos")
    parser.add_argument("--fade-duration", type=float, default=1.0, help="Fade in/out duration in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    bg_color = tuple(map(int, args.bg_color.split(',')))
    highlight_color = tuple(map(int, args.highlight_color.split(',')))
    text_color = tuple(map(int, args.text_color.split(',')))
    headline = args.headline
    
    broll_map = None
    if args.broll_dir and os.path.isdir(args.broll_dir):
        broll_map = {}
        for fname in os.listdir(args.broll_dir):
            key = os.path.splitext(fname)[0].lower()
            broll_map[key] = os.path.join(args.broll_dir, fname)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "transcricoes"), exist_ok=True)
    
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    log_path = os.path.join(logs_dir, "processamento.log")
    
    # configure global logger path
    global LOG_PATH
    LOG_PATH = log_path
    
    log(f"Log started at {time.asctime()}")
    
    # Verificar se arquivo de vídeo existe
    if not os.path.exists(args.video_path):
        log(f"ERRO: Arquivo de vídeo não encontrado: {args.video_path}")
        return
    
    audio_path = extract_audio(args.video_path)
    log(f"Audio extracted to {audio_path}")
    
    transcription_segments = transcribe_audio(audio_path, args.whisper_model)
    
    if not args.no_review:
        transcription_segments = review_transcription(transcription_segments)
    
    txt_path = os.path.join(args.output_dir, "transcricoes", "completa.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(seg.get("text","") for seg in transcription_segments))
    
    with open(os.path.join(args.output_dir, "transcricoes", "segmentos.json"), "w", encoding="utf-8") as f:
        json.dump(transcription_segments, f, indent=2)
    
    clip_finder = LLMClipFinder(api_key=args.api_key)
    clip_suggestions = clip_finder.find_interesting_moments(transcription_segments, args.min_clips, args.max_clips)
    log(f"Initial clip suggestions: {clip_suggestions}")
    
    viral_intervals = detectar_momentos_virais(audio_path, transcription_segments)
    
    if viral_intervals:
        log(f"[debug] viral intervals detected: {viral_intervals}")
        if not clip_suggestions or "clips" not in clip_suggestions:
            clip_suggestions = {"clips": []}
        
        for vi in viral_intervals:
            clip_suggestions.setdefault("clips", []).append({
                "start": f"{int(vi['start']//60):02d}:{int(vi['start']%60):02d}",
                "end": f"{int(vi['end']//60):02d}:{int(vi['end']%60):02d}",
                "reason": vi.get("keyword","viral moment"),
                "viral_score": vi.get("score", 1.0),
                "viral_source": vi.get("source", "audio")
            })
    
    if not clip_suggestions or "clips" not in clip_suggestions or not clip_suggestions["clips"]:
        log("No clip suggestions obtained; falling back to uniform segmentation")
        clip_suggestions = {"clips": []}
        total_dur = transcription_segments[-1].get("end", 0) if transcription_segments else 0
        for i in range(args.min_clips):
            s = min(i * 30, max(0, total_dur - 15))
            e = min(s + 30, total_dur)
            clip_suggestions["clips"].append({
                "start": f"{int(s//60):02d}:{int(s%60):02d}",
                "end": f"{int(e//60):02d}:{int(e%60):02d}",
                "reason": "fallback",
                "viral_score": 0.5
            })
    
    # enforce at least one hook early
    if clip_suggestions.get("clips"):
        if all(parse_timestamp(c["start"]) > 3 for c in clip_suggestions["clips"]):
            log("adding early hook clip")
            clip_suggestions["clips"].insert(0,{
                "start":"00:00",
                "end":"00:15",
                "reason":"hook",
                "viral_score": 1.0
            })
    
    # clamp durations to 15-45s
    for c in clip_suggestions.get("clips",[]):
        s = parse_timestamp(c["start"])
        e = parse_timestamp(c["end"])
        if e - s < 15:
            e = s + 15
        if e - s > 45:
            e = s + 45
        c["start"] = f"{int(s//60):02d}:{int(s%60):02d}"
        c["end"] = f"{int(e//60):02d}:{int(e%60):02d}"
    
    created_clips = []
    clip_texts = []
    clipes_info_para_ranking = []
    
    for i, clip in enumerate(clip_suggestions["clips"]):
        start = clip.get("start")
        end = clip.get("end")
        caption = clip.get("caption", "")
        
        clip_obj = {"start": start, "end": end, "caption": caption, "segments": []}
        
        # collect word segments overlapping
        for seg in transcription_segments:
            if seg['end'] >= parse_timestamp(start) and seg['start'] <= parse_timestamp(end):
                import copy
                clip_obj["segments"].append(copy.deepcopy(seg))
        
        # Extrair texto completo do clipe
        clip_texto = " ".join([seg.get("text","") for seg in clip_obj.get("segments", [])])
        clip_texts.append(clip_texto)
        
        # Coletar informações para ranking
        clip_info = {
            "id": i + 1,
            "start": start,
            "end": end,
            "start_time": parse_timestamp(start),
            "end_time": parse_timestamp(end),
            "duracao": parse_timestamp(end) - parse_timestamp(start),
            "texto": clip_texto,
            "palavras_chave": [kw for kw in clip_texto.lower().split() if len(kw) > 3][:10],
            "viral_score": clip.get("viral_score", 0.5),
            "viral_source": clip.get("viral_source", "unknown"),
            "tem_risada": False,
            "tem_pergunta": any(p in clip_texto for p in ['?', 'como', 'quando', 'onde', 'por que'])
        }
        
        # Verificar se tem risada no intervalo
        for vi in viral_intervals:
            if vi.get("source") == "laughter":
                vi_start = vi.get("start", 0)
                vi_end = vi.get("end", 0)
                if (vi_start <= clip_info["end_time"] and vi_end >= clip_info["start_time"]):
                    clip_info["tem_risada"] = True
                    clip_info["viral_score"] = max(clip_info["viral_score"], vi.get("score", 2.5))
                    break
        
        clipes_info_para_ranking.append(clip_info)
    
    # RANKING DOS CLIPES POR IA
    if args.rank_clips:
        log("\n" + "="*50)
        log("GERANDO RANKING DOS CLIPES POR IA")
        log("="*50)
        
        clipes_ordenados = rankear_clipes_por_ia(clipes_info_para_ranking)
        
        # Mostrar ranking completo
        log("\n" + "="*50)
        log("RANKING FINAL DOS CLIPES:")
        log("="*50)
        for clip in clipes_ordenados:
            estrelas = '★' * int(clip.get('nota_ia', 0) / 2) + '☆' * (5 - int(clip.get('nota_ia', 0) / 2))
            log(f"#{clip['ranking']} - Clip {clip['id']} | NOTA: {clip['nota_ia']}/10 {estrelas}")
            log(f"   Texto: {clip['texto'][:100]}...")
            if clip.get('justificativa_ia'):
                log(f"   Justificativa: {clip['justificativa_ia']}")
            log("")
        
        # Salvar ranking em arquivo
        ranking_path = os.path.join(args.output_dir, "ranking_clipes.json")
        with open(ranking_path, "w", encoding="utf-8") as f:
            json.dump(clipes_ordenados, f, indent=2, ensure_ascii=False)
        log(f"Ranking salvo em: {ranking_path}")
    
    # CRIAR CLIPS (na ordem original, não na ordem do ranking)
    for i, clip in enumerate(clip_suggestions["clips"]):
        clip_obj = {"start": clip["start"], "end": clip["end"], "caption": clip.get("caption", ""), "segments": []}
        
        # Re-coletar segments (já que perdemos referência)
        for seg in transcription_segments:
            if seg['end'] >= parse_timestamp(clip["start"]) and seg['start'] <= parse_timestamp(clip["end"]):
                import copy
                clip_obj["segments"].append(copy.deepcopy(seg))
        
        # Encontrar ranking para este clipe (se disponível)
        ranking_info = None
        if args.rank_clips and clipes_ordenados:
            for rc in clipes_ordenados:
                if rc['id'] == i + 1:
                    ranking_info = rc
                    break
        
        output_path = os.path.join(args.output_dir, f"clip_{i+1}")
        
        add_ranking_overlay = args.rank_clips and not args.no_ranking_overlay
        
        clip_path = create_clip(
            args.video_path,
            clip_obj,
            output_path,
            captions=args.captions,
            bg_color=bg_color,
            highlight_color=highlight_color,
            text_color=text_color,
            headline_text=headline,
            broll_map=broll_map,
            remove_silence=True,
            retranscribe=args.retranscribe,
            ranking_info=ranking_info if add_ranking_overlay else None,
            fade_duration=args.fade_duration
        )
        
        if clip_path:
            created_clips.append(clip_path)
            log(f"Successfully created clip at {clip_path}")
        else:
            log(f"Failed to create clip {i+1}")
    
    if created_clips:
        # Usar clipes ordenados para estrutura se disponível
        rankings_para_estrutura = clipes_ordenados if args.rank_clips else None
        criar_estrutura_clipes(created_clips, clip_texts, base_dir=args.output_dir, rankings=rankings_para_estrutura)
    
    try:
        os.remove(audio_path)
    except Exception:
        pass
    
    log(f"\n" + "="*50)
    log(f"PROCESSO COMPLETO!")
    log(f"="*50)
    log(f"Total de clipes criados: {len(created_clips)}")
    log(f"Diretório de saída: {args.output_dir}")
    
    if args.rank_clips and clipes_ordenados:
        log(f"\nRESUMO DO RANKING:")
        for clip in clipes_ordenados:
            estrelas = '★' * int(clip.get('nota_ia', 0) / 2) + '☆' * (5 - int(clip.get('nota_ia', 0) / 2))
            log(f"Clip {clip['id']}: {clip['nota_ia']}/10 {estrelas}")

if __name__ == "__main__":
    main()