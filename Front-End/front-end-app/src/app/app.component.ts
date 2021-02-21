import { HttpResponse, HttpErrorResponse } from '@angular/common/http';
import { Component, Input } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { AlertDialogComponent } from './alert-dialog/alert-dialog.component';
import { Diagnose } from './models/Diagnose';
import { MachineLearningApiService } from './services/machine-learning-api.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  title = 'front-end-app';
  @Input() diagnose: Diagnose = {};

  constructor(
    private api: MachineLearningApiService,
    private dialog: MatDialog
  ) {}

  public getDiagnose(): void {
    console.log(this.diagnose);
    this.api.getDiagnose(this.diagnose).subscribe((diagnose_result) => {
      console.log(diagnose_result);
      this.showDiagnoseResult(diagnose_result);
    });
  }

  private showDiagnoseResult(diagnose_result: number): void {
    let dialogRef = this.dialog.open(AlertDialogComponent, {
      data: { dataKey: diagnose_result },
    });
    dialogRef.afterClosed().subscribe((result) => {
      if (result == 'confirm') {
        console.log('Diagnose Confirmed.');
      }
    });
  }
}
