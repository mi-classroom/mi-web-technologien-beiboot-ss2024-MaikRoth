// WebSocketService
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import io from 'socket.io-client';

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {
  private socket;

  constructor() {
    this.socket = io('http://localhost:3000');
  }

  public onImageReady(): Observable<string> {
    return new Observable<string>(subscriber => {
      this.socket.on('imageReady', (data) => {
        console.log('Received image data:', data);
        subscriber.next(`http://localhost:3000/outputs/${data.imageUrl}`);
      });
    });
  }
  public onProcessingProgress(): Observable<number> {
    return new Observable<number>(subscriber => {
        this.socket.on('processingProgress', (data) => {
            console.log('Received progress:', data.progress);
            subscriber.next(data.progress);
        });
    });
}
}
