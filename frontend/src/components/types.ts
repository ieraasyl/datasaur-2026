export interface Diagnosis {
  rank: number;
  diagnosis: string;
  icd10_code: string;
  explanation: string;
}

export interface DiagnoseResponse {
  diagnoses: Diagnosis[];
}
