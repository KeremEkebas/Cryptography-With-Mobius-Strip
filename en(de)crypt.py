#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import hashlib, json, base64, os, math, random, time, hmac, secrets
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

class MobiusMap:
    def __init__(self, radius: float = 1.0, width: float = 0.5):
        self.radius = radius
        self.width = width
        
        self.u = 0.0
        self.v = 0.0
        
        self.grid_size = 50
        self.create_uv_grid()
    
    def create_uv_grid(self):
        self.u_grid = []
        self.v_grid = []
        self.points_3d = []
        
        for i in range(self.grid_size + 1):
            u_row = []
            v_row = []
            points_row = []
            for j in range(self.grid_size + 1):
                u = (i / self.grid_size) * 2 * math.pi
                v = (j / self.grid_size - 0.5) * self.width
                u_row.append(u)
                v_row.append(v)
                
                x, y, z = self.uv_to_3d(u, v)
                points_row.append((x, y, z))
            
            self.u_grid.append(u_row)
            self.v_grid.append(v_row)
            self.points_3d.append(points_row)
    
    def uv_to_3d(self, u: float, v: float) -> Tuple[float, float, float]:
        half_u = u * 0.5
        
        x = (self.radius + v * math.cos(half_u)) * math.cos(u)
        y = (self.radius + v * math.cos(half_u)) * math.sin(u)
        z = v * math.sin(half_u)
        
        return x, y, z
    
    def get_normal(self, u: float, v: float) -> Tuple[float, float, float]:
        half_u = u * 0.5
        
        cos_half_u = math.cos(half_u)
        sin_half_u = math.sin(half_u)
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        
        dx_du = -(self.radius + v * cos_half_u) * sin_u - v * 0.5 * sin_half_u * cos_u
        dy_du = (self.radius + v * cos_half_u) * cos_u - v * 0.5 * sin_half_u * sin_u
        dz_du = v * 0.5 * cos_half_u
        
        dx_dv = cos_half_u * cos_u
        dy_dv = cos_half_u * sin_u
        dz_dv = sin_half_u
        
        nx = dy_du * dz_dv - dz_du * dy_dv
        ny = dz_du * dx_dv - dx_du * dz_dv
        nz = dx_du * dy_dv - dy_du * dx_dv
        
        norm = math.sqrt(nx*nx + ny*ny + nz*nz)
        if norm > 0:
            nx /= norm
            ny /= norm
            nz /= norm
        
        return nx, ny, nz
    
    def get_curvature(self, u: float, v: float) -> float:
        half_u = u * 0.5
        curvature = abs(math.sin(half_u)) * (1 + abs(v) / self.width)
        return curvature
    
    def walk_geodesic(self, steps: int = 10, step_size: float = 0.1):
        path = []
        for _ in range(steps):
            self.u = (self.u + step_size) % (2 * math.pi)
            self.v += 0.05 * math.sin(self.u * 2)
            
            half_width = self.width / 2
            if self.v < -half_width: self.v = -half_width
            elif self.v > half_width: self.v = half_width
            
            path.append((self.u, self.v))
        return path

class MobiusStrip:
    def __init__(self, radius: float = 1.0, width: float = 0.5):
        self.radius = radius
        self.width = width
        self.u = 0.0
        self.v = 0.0
        self.twist = 0.5
    
    def get_3d_point(self, u: Optional[float] = None, v: Optional[float] = None) -> Tuple[float, float, float]:
        if u is None: u = self.u
        if v is None: v = self.v
        
        cos_half_u = math.cos(self.twist * u)
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        sin_half_u = math.sin(self.twist * u)
        
        r_plus_v = self.radius + v * cos_half_u
        x = r_plus_v * cos_u
        y = r_plus_v * sin_u
        z = v * sin_half_u
        
        return x, y, z
    
    def walk(self, distance: float = 0.1) -> Tuple[float, float, float]:
        self.u = (self.u + distance) % (2 * math.pi)
        
        if self.u < distance:
            self.v = -self.v
        
        self.v += math.sin(self.u * 3) * 0.05
        half_width = self.width * 0.5
        if self.v < -half_width: self.v = -half_width
        elif self.v > half_width: self.v = half_width
        
        return self.get_3d_point()

def validate_hex_key(key: str) -> bool:
    if len(key) != 64: return False
    try: int(key, 16); return True
    except: return False

def generate_hex_key() -> str:
    return os.urandom(32).hex()

class SecureMobiusCipher:
    
    def __init__(self, key: str):
        if not validate_hex_key(key):
            raise ValueError("Anahtar 64 karakter hex (32 byte) olmalı")
        
        self.key = key
        self.key_bytes = bytes.fromhex(key)
        
        # HMAC için ayrı anahtar türet
        self.hmac_key = hashlib.sha256(self.key_bytes + b"HMAC_KEY_SALT").digest()
        
        self.strip = MobiusStrip(
            radius=1.0 + (self.key_bytes[0] / 255.0) * 0.5,
            width=0.3 + (self.key_bytes[1] / 255.0) * 0.4
        )
        
        self.start_u = (self.key_bytes[2] / 255.0) * 2 * math.pi
        self.start_v = ((self.key_bytes[3] / 255.0) - 0.5) * self.strip.width
        
        self.strip.u = self.start_u
        self.strip.v = self.start_v
        
        self.mapper = MobiusMap(self.strip.radius, self.strip.width)
        self.mapper.u = self.start_u
        self.mapper.v = self.start_v
    
    def _safe_transform(self, byte_val: int, x: float, y: float, z: float, forward: bool = True) -> int:
        geo_val = int((abs(x) + abs(y) + abs(z)) * 100) % 256
        angle_val = int(self.strip.u * 100) % 256
        pos_val = int((self.strip.v + self.strip.width/2) * 100) % 256
        
        if forward:
            result = byte_val
            result ^= geo_val
            result = (result + angle_val) % 256
            rotate = (geo_val % 7) + 1
            result = ((result << rotate) | (result >> (8 - rotate))) & 0xFF
            result ^= pos_val
        else:
            result = byte_val
            result ^= pos_val
            rotate = (geo_val % 7) + 1
            result = ((result >> rotate) | (result << (8 - rotate))) & 0xFF
            result = (result - angle_val) % 256
            result ^= geo_val
        
        return result
    
    def _mobius_operation(self, data: bytes, encrypt: bool = True) -> bytes:
        if len(data) < 2:
            return data
        
        self.strip.u = self.start_u
        self.strip.v = self.start_v
        
        if encrypt:
            mid = len(data) // 2
            part1 = data[:mid]
            part2 = data[mid:]
            
            transformed2 = bytearray()
            for byte in part2:
                x, y, z = self.strip.walk(0.05)
                transformed2.append(self._safe_transform(byte, x, y, z, True))
            
            return bytes(transformed2) + part1
        else:
            self.strip.u = self.start_u
            self.strip.v = self.start_v
            
            transformed_len = len(data) // 2
            part2_encrypted = data[:len(data) - transformed_len]
            part1 = data[len(data) - transformed_len:]
            
            part2 = bytearray()
            for byte in part2_encrypted:
                x, y, z = self.strip.walk(0.05)
                part2.append(self._safe_transform(byte, x, y, z, False))
            
            return part1 + bytes(part2)
    
    def _layer_transform(self, data: bytes, encrypt: bool = True) -> bytes:
        result = bytearray()
        
        self.strip.u = self.start_u
        self.strip.v = self.start_v
        
        for byte in data:
            x, y, z = self.strip.walk(0.1)
            result.append(self._safe_transform(byte, x, y, z, encrypt))
        
        return bytes(result)
    
    def encrypt(self, plaintext: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        # Nonce (rastgele değer) ekle - determinizmi kır
        nonce = secrets.token_bytes(16)  # 128-bit nonce
        timestamp = int(time.time() * 1000)  # milisaniye
        
        # Nonce ve timestamp'i plaintext'ten önce ekle
        plaintext_bytes = plaintext.encode('utf-8')
        data_with_metadata = nonce + timestamp.to_bytes(8, 'big') + plaintext_bytes
        
        data = data_with_metadata
        
        # 3 katmanlı şifreleme
        for layer in range(3):
            data = self._mobius_operation(data, True)
            data = self._layer_transform(data, True)
            self.strip.u = self.start_u
            self.strip.v = self.start_v
        
        execution_time = time.perf_counter() - start_time
        ciphertext_b64 = base64.b64encode(data).decode('ascii')
        
        # HMAC hesapla (ciphertext üzerinden)
        hmac_digest = hmac.new(self.hmac_key, data, hashlib.sha256).hexdigest()
        
        mapping_info = {
            "grid_size": self.mapper.grid_size,
            "current_u": self.mapper.u,
            "current_v": self.mapper.v,
            "curvature": self.mapper.get_curvature(self.mapper.u, self.mapper.v)
        }
        
        return {
            "ciphertext": ciphertext_b64,
            "params": {
                "start_u": self.start_u,
                "start_v": self.start_v,
                "radius": self.strip.radius,
                "width": self.strip.width,
                "key_prefix": self.key[:16],
                "time": f"{execution_time:.4f}s",
                "length": len(plaintext),
                "algorithm": "SecureMobius",
                "version": "3.0",
                "layers": 3,
                "mapping": mapping_info,
                "nonce": nonce.hex(),
                "timestamp": timestamp
            },
            "signature": hmac_digest
        }
    
    def decrypt(self, encrypted_packet: Dict[str, Any]) -> str:
        try:
            start_time = time.perf_counter()
            
            if isinstance(encrypted_packet, str):
                try:
                    encrypted_packet = json.loads(encrypted_packet)
                except:
                    encrypted_packet = {"ciphertext": encrypted_packet}
            
            ciphertext = encrypted_packet.get("ciphertext")
            if not ciphertext:
                raise ValueError("Ciphertext bulunamadı")
            
            data = base64.b64decode(ciphertext)
            params = encrypted_packet.get("params", {})
            
            # HMAC doğrulama (eğer varsa)
            if "signature" in encrypted_packet:
                expected_hmac = hmac.new(self.hmac_key, data, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected_hmac, encrypted_packet["signature"]):
                    raise ValueError("İmza doğrulaması başarısız - veri manipüle edilmiş olabilir")
            
            # Parametreleri ayarla
            if "start_u" in params:
                self.start_u = float(params["start_u"])
            if "start_v" in params:
                self.start_v = float(params["start_v"])
            
            self.strip.u = self.start_u
            self.strip.v = self.start_v
            
            # 3 katmanlı çözme
            for layer in range(3):
                data = self._layer_transform(data, False)
                data = self._mobius_operation(data, False)
                self.strip.u = self.start_u
                self.strip.v = self.start_v
            
            # Nonce ve timestamp'i çıkar (ilk 24 byte: nonce(16) + timestamp(8))
            if len(data) >= 24:
                # İlk 24 byte'ı at, kalanı plaintext
                plaintext_bytes = data[24:]
            else:
                plaintext_bytes = data
            
            try:
                plaintext = plaintext_bytes.decode('utf-8')
            except UnicodeDecodeError:
                plaintext = plaintext_bytes.decode('utf-8', errors='replace')
            
            execution_time = time.perf_counter() - start_time
            
            return plaintext
            
        except Exception as e:
            raise Exception(f"Çözme hatası: {str(e)}")

class MobiusMapVisualizer:
    
    def __init__(self, parent, width=400, height=400):
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='#0a0a1a')
        self.width = width
        self.height = height
        
        self.colors = {
            'bg': '#0a0a1a',
            'grid': '#1e1b4b',
            'uv_grid': '#4f46e5',
            'point': '#10b981',
            'path': '#f87171',
            'text': '#e2e8f0',
            'axis': '#3b82f6'
        }
        
        self.current_map = None
        self.path_points = []
        
        self._draw_base()
    
    def _draw_base(self):
        grid_size = 40
        for x in range(0, self.width, grid_size):
            self.canvas.create_line(x, 0, x, self.height, 
                                  fill=self.colors['grid'], width=1)
        for y in range(0, self.height, grid_size):
            self.canvas.create_line(0, y, self.width, y,
                                  fill=self.colors['grid'], width=1)
        
        center_x, center_y = self.width//2, self.height//2
        self.canvas.create_line(center_x, 0, center_x, self.height,
                              fill=self.colors['axis'], width=2)
        self.canvas.create_line(0, center_y, self.width, center_y,
                              fill=self.colors['axis'], width=2)
        
        self.canvas.create_text(center_x + 10, 20, text="U (0-2π)", 
                               fill=self.colors['text'], font=("Arial", 9))
        self.canvas.create_text(20, center_y - 10, text="V (-w/2 to w/2)", 
                               fill=self.colors['text'], font=("Arial", 9))
    
    def draw_map(self, mobius_map: MobiusMap):
        self.current_map = mobius_map
        self.canvas.delete("map")
        
        u_scale = self.width / (2 * math.pi)
        v_scale = self.height / mobius_map.width
        
        for i in range(mobius_map.grid_size):
            for j in range(mobius_map.grid_size):
                u1 = mobius_map.u_grid[i][j]
                v1 = mobius_map.v_grid[i][j]
                u2 = mobius_map.u_grid[i+1][j]
                v2 = mobius_map.v_grid[i+1][j]
                u3 = mobius_map.u_grid[i+1][j+1]
                v3 = mobius_map.v_grid[i+1][j+1]
                u4 = mobius_map.u_grid[i][j+1]
                v4 = mobius_map.v_grid[i][j+1]
                
                points = []
                for u, v in [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]:
                    px = u * u_scale
                    py = self.height/2 + v * v_scale
                    points.extend([px, py])
                
                self.canvas.create_polygon(points, outline=self.colors['uv_grid'], 
                                         fill='', width=1, tags="map")
        
        self.draw_point(mobius_map.u, mobius_map.v, "Başlangıç")
    
    def draw_point(self, u: float, v: float, label: str = ""):
        if not self.current_map:
            return
        
        u_scale = self.width / (2 * math.pi)
        v_scale = self.height / self.current_map.width
        
        px = u * u_scale
        py = self.height/2 + v * v_scale
        
        self.canvas.create_oval(px-6, py-6, px+6, py+6,
                              fill=self.colors['point'], outline='white',
                              width=2, tags="map")
        
        self.path_points.append((px, py))
        
        self.canvas.delete("path")
        if len(self.path_points) > 1:
            points = []
            for x, y in self.path_points:
                points.extend([x, y])
            
            self.canvas.create_line(*points, fill=self.colors['path'], 
                                  width=2, smooth=True, tags="path")
        
        if label:
            self.canvas.create_text(px + 15, py - 15, text=label,
                                  fill=self.colors['text'], font=("Arial", 8),
                                  tags="map")
    
    def draw_geodesic_path(self, mobius_map: MobiusMap, steps: int = 20):
        path = mobius_map.walk_geodesic(steps)
        for u, v in path:
            self.draw_point(u, v)
    
    def clear_path(self):
        self.path_points = []
        self.canvas.delete("path")
        self.canvas.delete("point")
    
    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)

class MobiusCryptoProGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🗺️ Möbius Strip Şifreleme Pro")
        self.root.geometry("1400x850")
        
        self.bg = '#0a0a1a'
        self.sidebar_bg = '#1e1b4b'
        self.card_bg = '#312e81'
        self.text = '#e2e8f0'
        self.accent = '#4f46e5'
        
        self.root.configure(bg=self.bg)
        self.create_widgets()
        self.set_defaults()
    
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg=self.bg)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        top_panel = tk.Frame(main_frame, bg=self.bg)
        top_panel.pack(fill='x', pady=(0, 20))
        
        left_panel = tk.Frame(top_panel, bg=self.sidebar_bg, width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 20))
        left_panel.pack_propagate(False)
        
        self.create_left_panel(left_panel)
        
        right_panel = tk.Frame(top_panel, bg=self.bg)
        right_panel.pack(side='right', fill='both', expand=True)
        
        strip_frame = tk.LabelFrame(right_panel, text=" 🌀 3D MÖBİUS STRIP ",
                                   bg=self.bg, fg=self.text,
                                   font=('Segoe UI', 11, 'bold'))
        strip_frame.pack(fill='x', pady=(0, 10))
        
        self.strip_canvas = tk.Canvas(strip_frame, width=500, height=250, bg='#0a0a1a')
        self.strip_canvas.pack(padx=10, pady=10)
        
        map_frame = tk.LabelFrame(right_panel, text=" 🗺️ UV HARİTALANDIRMA ",
                                 bg=self.bg, fg=self.text,
                                 font=('Segoe UI', 11, 'bold'))
        map_frame.pack(fill='x')
        
        self.map_viz = MobiusMapVisualizer(map_frame, width=500, height=250)
        self.map_viz.pack(padx=10, pady=10)
        
        bottom_panel = tk.Frame(main_frame, bg=self.bg)
        bottom_panel.pack(fill='both', expand=True)
        
        button_frame = tk.Frame(bottom_panel, bg=self.bg)
        button_frame.pack(fill='x', pady=(0, 10))
        
        buttons = [
            ("🔒 Şifrele", self.encrypt, '#4f46e5'),
            ("🔓 Çöz", self.decrypt, '#10b981'),
            ("🗺️ Haritayı Göster", self.show_map, '#8b5cf6'),
            ("🌀 UV Yürüyüşü", self.walk_uv, '#f59e0b'),
            ("🧪 Test Et", self.run_test, '#ec4899'),
            ("🗑️ Temizle", self.clear_all, '#ef4444')
        ]
        
        for text, cmd, color in buttons:
            btn = tk.Button(button_frame, text=text, command=cmd,
                          bg=color, fg='white', font=('Segoe UI', 10, 'bold'),
                          padx=15, pady=8)
            btn.pack(side='left', padx=5)
        
        notebook = ttk.Notebook(bottom_panel)
        notebook.pack(fill='both', expand=True)
        
        input_frame = tk.Frame(notebook, bg=self.bg)
        notebook.add(input_frame, text="📝 Giriş")
        
        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD,
                                                  font=('Consolas', 11),
                                                  bg=self.card_bg, fg=self.text,
                                                  height=12)
        self.input_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        output_frame = tk.Frame(notebook, bg=self.bg)
        notebook.add(output_frame, text="📤 Çıkış")
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD,
                                                   font=('Consolas', 11),
                                                   bg=self.card_bg, fg='#fbbf24',
                                                   height=12)
        self.output_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        json_frame = tk.Frame(notebook, bg=self.bg)
        notebook.add(json_frame, text="{} JSON")
        
        self.json_text = scrolledtext.ScrolledText(json_frame, wrap=tk.WORD,
                                                 font=('Consolas', 10),
                                                 bg=self.card_bg, fg='#34d399',
                                                 height=12)
        self.json_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="✅ Sistem hazır")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                            bg=self.sidebar_bg, fg=self.text,
                            font=('Segoe UI', 10), anchor='w')
        status_bar.pack(side='bottom', fill='x', padx=20, pady=(0, 10))
    
    def create_left_panel(self, parent):
        tk.Label(parent, text="🗺️", font=('Arial', 40), 
                bg=self.sidebar_bg, fg=self.accent).pack(pady=(20, 5))
        tk.Label(parent, text="MOBİUS PRO", font=('Segoe UI', 18, 'bold'),
                bg=self.sidebar_bg, fg=self.text).pack()
        tk.Label(parent, text="Şifreleme & Haritalandırma",
                bg=self.sidebar_bg, fg='#94a3b8').pack(pady=(0, 20))
        
        key_frame = tk.LabelFrame(parent, text=" 🔑 64-HEX ANAHTAR ",
                                 bg=self.sidebar_bg, fg=self.text,
                                 font=('Segoe UI', 11, 'bold'))
        key_frame.pack(fill='x', padx=20, pady=10)
        
        self.key_var = tk.StringVar()
        self.key_var.trace('w', self.on_key_change)
        
        key_entry = tk.Entry(key_frame, textvariable=self.key_var,
                          font=('Consolas', 10), bg=self.card_bg,
                          fg=self.text, insertbackground=self.text)
        key_entry.pack(fill='x', padx=10, pady=10)
        
        self.key_status = tk.Label(key_frame, text="❌ 0/64 karakter",
                                 bg=self.sidebar_bg, fg='#ef4444',
                                 font=('Segoe UI', 9))
        self.key_status.pack(anchor='w', padx=10, pady=(0, 10))
        
        key_btns = tk.Frame(key_frame, bg=self.sidebar_bg)
        key_btns.pack(fill='x', padx=10, pady=(0, 10))
        
        tk.Button(key_btns, text="🎲 Yeni", command=self.new_key,
                 bg=self.card_bg, fg=self.text, width=8).pack(side='left', padx=2)
        tk.Button(key_btns, text="📋 Kopyala", command=self.copy_key,
                 bg=self.card_bg, fg=self.text, width=8).pack(side='left', padx=2)
        tk.Button(key_btns, text="✓ Doğrula", command=self.validate_key,
                 bg=self.card_bg, fg=self.text, width=8).pack(side='left', padx=2)
        
        map_frame = tk.LabelFrame(parent, text=" ⚙️ HARİTALAMA ",
                                 bg=self.sidebar_bg, fg=self.text,
                                 font=('Segoe UI', 11, 'bold'))
        map_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Button(map_frame, text="🗺️ Haritayı Yenile", command=self.refresh_map,
                 bg=self.card_bg, fg=self.text).pack(fill='x', padx=10, pady=10)
        tk.Button(map_frame, text="🧹 Yolu Temizle", command=self.clear_path,
                 bg=self.card_bg, fg=self.text).pack(fill='x', padx=10, pady=(0, 10))
        
        info_frame = tk.LabelFrame(parent, text=" ℹ️ SİSTEM BİLGİSİ ",
                                 bg=self.sidebar_bg, fg=self.text,
                                 font=('Segoe UI', 11, 'bold'))
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = """• 64 hex anahtar (32 byte)
• UTF-8 uyumlu şifreleme
• 3 katmanlı işleme
• Möbius strip geometrisi
• UV haritalandırma
• Jeodezik yollar
• Gerçek zamanlı görselleştirme
• NONCE ile güvenlik
• HMAC ile manipülasyon koruması"""
        
        tk.Label(info_frame, text=info_text, bg=self.sidebar_bg,
                fg='#94a3b8', font=('Consolas', 9), justify='left').pack(padx=10, pady=10)
    
    def set_defaults(self):
        self.key_var.set(generate_hex_key())
        
        sample_text = "🗺️ MOBİUS STRIP PRO SİSTEMİ"


        
        self.input_text.insert('1.0', sample_text)
    
    def on_key_change(self, *args):
        key = self.key_var.get()
        if len(key) == 64 and validate_hex_key(key):
            self.key_status.config(text='✅ 64/64 karakter', fg='#10b981')
        else:
            self.key_status.config(text=f'❌ {len(key)}/64 karakter', fg='#ef4444')
    
    def new_key(self):
        self.key_var.set(generate_hex_key())
        self.status_var.set("✅ Yeni anahtar oluşturuldu")
    
    def copy_key(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.key_var.get())
        self.status_var.set("✅ Anahtar panoya kopyalandı")
    
    def validate_key(self):
        key = self.key_var.get()
        if validate_hex_key(key):
            messagebox.showinfo('Geçerli', '✅ Anahtar geçerli!')
        else:
            messagebox.showerror('Geçersiz', '❌ Anahtar geçersiz!')
    
    def draw_strip(self, strip: MobiusStrip):
        self.strip_canvas.delete("all")
        
        center_x, center_y = 250, 125
        scale = 80
        
        points = []
        for i in range(80):
            u = (i / 80) * 2 * math.pi
            x, y, z = strip.get_3d_point(u, 0)
            px = center_x + x * scale
            py = center_y + z * scale
            points.append((px, py))
        
        if len(points) > 1:
            for i in range(len(points)-1):
                x1, y1 = points[i]
                x2, y2 = points[i+1]
                self.strip_canvas.create_line(x1, y1, x2, y2,
                                            fill='#4f46e5', width=3)
        
        x, y, z = strip.get_3d_point()
        px = center_x + x * scale
        py = center_y + z * scale
        
        self.strip_canvas.create_oval(px-8, py-8, px+8, py+8,
                                     fill='#10b981', outline='white', width=2)
        
        info = f"U: {strip.u:.2f} | V: {strip.v:.2f}"
        self.strip_canvas.create_text(10, 10, text=info, anchor='nw',
                                     fill='#e2e8f0', font=('Consolas', 9))
    
    def show_map(self):
        try:
            key = self.key_var.get()
            if not validate_hex_key(key):
                messagebox.showwarning('Hata', 'Geçerli bir anahtar girin!')
                return
            
            cipher = SecureMobiusCipher(key)
            self.map_viz.draw_map(cipher.mapper)
            self.draw_strip(cipher.strip)
            self.status_var.set("✅ Haritalandırma görselleştirildi")
            
        except Exception as e:
            messagebox.showerror('Hata', str(e))
    
    def walk_uv(self):
        try:
            key = self.key_var.get()
            if not validate_hex_key(key):
                messagebox.showwarning('Hata', 'Geçerli bir anahtar girin!')
                return
            
            if not hasattr(self, 'current_cipher'):
                self.current_cipher = SecureMobiusCipher(key)
            
            self.map_viz.draw_geodesic_path(self.current_cipher.mapper, 15)
            
            self.current_cipher.mapper.walk_geodesic(15)
            self.draw_strip(self.current_cipher.strip)
            
            self.status_var.set("✅ UV yürüyüşü tamamlandı")
            
        except Exception as e:
            messagebox.showerror('Hata', str(e))
    
    def refresh_map(self):
        self.show_map()
    
    def clear_path(self):
        self.map_viz.clear_path()
        self.status_var.set("✅ Yol temizlendi")
    
    def encrypt(self):
        try:
            key = self.key_var.get()
            if not validate_hex_key(key):
                messagebox.showwarning('Hata', 'Geçerli bir anahtar girin!')
                return
            
            plaintext = self.input_text.get('1.0', tk.END).strip()
            if not plaintext:
                messagebox.showwarning('Hata', 'Şifrelenecek metin girin!')
                return
            
            cipher = SecureMobiusCipher(key)
            result = cipher.encrypt(plaintext)
            
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert('1.0', result['ciphertext'])
            
            self.json_text.delete('1.0', tk.END)
            self.json_text.insert('1.0', json.dumps(result, indent=2))
            
            self.draw_strip(cipher.strip)
            self.map_viz.draw_map(cipher.mapper)
            
            self.status_var.set(f"✅ Şifreleme tamam! {len(plaintext)} karakter")
            
        except Exception as e:
            messagebox.showerror('Şifreleme Hatası', f'❌ Hata: {str(e)}')
    
    def decrypt(self):
        try:
            key = self.key_var.get()
            if not validate_hex_key(key):
                messagebox.showwarning('Hata', 'Geçerli bir anahtar girin!')
                return
            
            input_data = self.input_text.get('1.0', tk.END).strip()
            if not input_data:
                messagebox.showwarning('Hata', 'Çözülecek veri girin!')
                return
            
            packet = None
            if input_data.startswith('{') and input_data.endswith('}'):
                try:
                    packet = json.loads(input_data)
                except:
                    packet = {'ciphertext': input_data}
            else:
                packet = {'ciphertext': input_data}
            
            cipher = SecureMobiusCipher(key)
            plaintext = cipher.decrypt(packet)
            
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert('1.0', plaintext)
            
            self.draw_strip(cipher.strip)
            self.map_viz.draw_map(cipher.mapper)
            
            self.status_var.set(f"✅ Çözme tamam! {len(plaintext)} karakter")
            
        except Exception as e:
            messagebox.showerror('Çözme Hatası', f'❌ Hata: {str(e)}')
    
    def run_test(self):
        try:
            key = self.key_var.get()
            if not validate_hex_key(key):
                messagebox.showwarning('Hata', 'Geçerli bir anahtar girin!')
                return
            
            cipher = SecureMobiusCipher(key)
            
            test_cases = [
                "kerem bey",
                "Möbius strip haritalandırma testi",
                "İstanbul'da 3D geometri",
                "1234567890!@#$%^&*()_+-=",
                "Lorem ipsum dolor sit amet"
            ]
            
            results = []
            for text in test_cases:
                try:
                    encrypted = cipher.encrypt(text)
                    cipher2 = SecureMobiusCipher(key)
                    decrypted = cipher2.decrypt(encrypted)
                    results.append((text, text == decrypted))
                except Exception as e:
                    results.append((text, False, str(e)))
            
            result_text = "🧪 TEST SONUÇLARI\n" + "="*50 + "\n\n"
            success = 0
            
            for i, (text, ok, *error) in enumerate(results, 1):
                if ok:
                    result_text += f"✅ Test {i}: '{text[:20]}...' BAŞARILI\n"
                    success += 1
                else:
                    result_text += f"❌ Test {i}: '{text[:20]}...' HATALI"
                    if error:
                        result_text += f" ({error[0]})\n"
                    else:
                        result_text += "\n"
                       
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert('1.0', result_text)
            
            messagebox.showinfo('Test Tamamlandı',
                              f'{success}/{len(test_cases)} test başarılı!\nNon-deterministic: {unique}/5')
            
        except Exception as e:
            messagebox.showerror('Test Hatası', f'❌ Hata: {str(e)}')
    
    def clear_all(self):
        self.input_text.delete('1.0', tk.END)
        self.output_text.delete('1.0', tk.END)
        self.json_text.delete('1.0', tk.END)
        self.strip_canvas.delete('all')
        self.map_viz.canvas.delete('all')
        self.map_viz._draw_base()
        self.status_var.set("✅ Temizlendi")

def main():
    root = tk.Tk()
    root.title("🗺️ Möbius Strip Şifreleme Pro")
    
    root.update_idletasks()
    width = 1400
    height = 850
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    app = MobiusCryptoProGUI(root)
    root.mainloop()





if __name__ == "__main__":
    main()
