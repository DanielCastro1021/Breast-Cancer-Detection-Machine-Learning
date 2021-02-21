import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { HttpClient, HttpResponse } from '@angular/common/http';
import { Diagnose } from '../models/Diagnose';

@Injectable({
  providedIn: 'root',
})
export class MachineLearningApiService {
  API_URL = 'http://127.0.0.1:8000';
  constructor(protected http: HttpClient) {}

  public getDiagnose(diagnose: Diagnose): Observable<any> {
    return this.http.post<any>(this.API_URL + '/diagnose_predict', diagnose);
  }
}
