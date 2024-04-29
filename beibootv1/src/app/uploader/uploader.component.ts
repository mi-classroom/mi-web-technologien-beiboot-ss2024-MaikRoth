import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { NgZone } from '@angular/core';
import { WebSocketService } from '../web-socket.service';

@Component({
  selector: 'app-uploader',
  templateUrl: './uploader.component.html',
  styleUrls: ['./uploader.component.css']
})
export class UploaderComponent {
  imageUrl: string;
  selectedFile: File = null;
  progress = 0;
  fps: number;
  windowSize: number;
  useWindow: boolean;

  constructor(private zone: NgZone, private webSocketService: WebSocketService, private http: HttpClient) {
    this.webSocketService.onImageReady().subscribe((data: any) => {
      setTimeout(() => {
        console.log("Received image data:", data);
        if (data) {
          this.imageUrl = `http://localhost:3000/outputs/${data}`;
        } else {
          console.error("Invalid image URL received:", data);
        }
      }, 4000);
    });
    this.webSocketService.onProcessingProgress().subscribe((progress: number) => {
      this.zone.run(() => {
        this.progress = progress;
      });
    });
  }

  onFileSelected(event) {
    this.selectedFile = event.target.files[0];
  }

  uploadFile() {
    if (!this.selectedFile || !this.fps || (this.useWindow && !this.windowSize)) {
      alert('Please select a file and specify FPS and window size.');
      return;
    }
    const formData = new FormData();
    formData.append('useWindow', this.useWindow ? 'true' : 'false');
    formData.append('video', this.selectedFile, this.selectedFile.name);
    formData.append('fps', this.fps.toString());
    if (this.useWindow) {
      formData.append('windowSize', this.windowSize.toString());
    }
    console.log('Uploading file:', this.selectedFile.name, 'with FPS:', this.fps, 'and window size:', this.windowSize);

    this.http.post('http://localhost:3000/upload', formData).subscribe({
      next: (response) => console.log('Upload started:', response),
      error: (error) => console.error('Error:', error)
    });
  }
}
