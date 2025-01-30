
export interface IQuantumProvider {
    id: string;
    name: string;
    qubits: number;
    submit(circuit: QuantumCircuit): Promise<ExecutionResult>;
    getStatus(): Promise<ProviderStatus>;
    calibrate(): Promise<CalibrationMetrics>;
}

export interface ProviderStatus {
    available: boolean;
    queueDepth: number;
    uptime: number;
    errorRates: Map<string, number>;
}

export type ExecutionResult = {
    measurements: number[];
    executionTime: number;
    error?: string;
};