import { TestBed } from '@angular/core/testing';

import { MachineLearningApiService } from './machine-learning-api.service';

describe('MachineLearningApiService', () => {
  let service: MachineLearningApiService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(MachineLearningApiService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
