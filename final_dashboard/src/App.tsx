import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Overview from './pages/Overview';
import LivePredictions from './pages/LivePredictions';
import DriftMonitoring from './pages/DriftMonitoring'; // <--- Import this
import ModelComparison from './pages/ModelComparison';
import Retrain from './pages/Retrain';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="live" element={<LivePredictions />} />
          <Route path="drift" element={<DriftMonitoring />} /> {/* <--- UPDATE THIS LINE */}
          <Route path="compare" element={<ModelComparison />} />
          <Route path="retrain" element={<Retrain />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;