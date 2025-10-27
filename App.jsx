import React, {useState} from 'react'
import axios from 'axios'

const API_ROOT = import.meta.env.VITE_API_ROOT || 'http://localhost:5000'

export default function App(){
  const [q, setQ] = useState('AAPL')
  const [results, setResults] = useState([])
  const [selected, setSelected] = useState(null)
  const [prediction, setPrediction] = useState(null)

  async function search(){
    const r = await axios.get(`${API_ROOT}/api/search?q=${encodeURIComponent(q)}`)
    setResults(r.data)
  }
  async function select(sym){
    setSelected(sym)
    setPrediction(null)
  }
  async function runPredict(timeframe='day'){
    const r = await axios.post(`${API_ROOT}/api/predict`, { ticker: selected, timeframe })
    setPrediction(r.data)
  }

  return (
    <div style={{padding:20,fontFamily:'Inter, system-ui'}}>
      <h1>day trdr (demo)</h1>
      <div style={{display:'flex',gap:10}}>
        <input value={q} onChange={e=>setQ(e.target.value)} />
        <button onClick={search}>Search</button>
      </div>
      <div style={{display:'flex',gap:20,marginTop:20}}>
        <aside style={{width:300}}>
          <h3>Results</h3>
          {results.map(r=> <div key={r.symbol}><button onClick={()=>select(r.symbol)}>{r.symbol} — {r.name}</button></div>)}
        </aside>
        <main style={{flex:1}}>
          {selected ? (
            <>
              <h2>{selected}</h2>
              <button onClick={()=>runPredict('scalp')}>Predict (scalp)</button>
              <button onClick={()=>runPredict('day')}>Predict (day)</button>
              <button onClick={()=>runPredict('swing')}>Predict (swing)</button>
              {prediction && <pre style={{background:'#f6f8fa',padding:12}}>{JSON.stringify(prediction,null,2)}</pre>}
            </>
          ) : <div>Select a symbol</div>}
        </main>
      </div>
    </div>
  )
}
