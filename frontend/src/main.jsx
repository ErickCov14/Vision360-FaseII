import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const clases = ['Grieta', 'Porosidad', 'Falta de penetración', 'Sin defecto']
function App() {
  const [archivo, setArchivo] = useState(null)
  const [vista, setVista] = useState(null)
  const [resultado, setResultado] = useState(null)
  const [error, setError] = useState('')
  const [cargando, setCargando] = useState(false)
  const [estado, setEstado] = useState('comprobando')
  const [info, setInfo] = useState(null)
  const entrada = useRef(null)

  useEffect(() => {
    fetch('/api/health').then(r => { if (!r.ok) throw Error('No disponible'); return r.json() })
      .then(() => setEstado('activo')).catch(() => setEstado('sin conexión'))
    fetch('/api/model').then(r => r.json()).then(setInfo).catch(() => {})
  }, [])

  function cambiarArchivo(f) {
    setError(''); setResultado(null)
    if (!f) return
    if (f.size > 8 * 1024 * 1024) { setError('Archivo demasiado grande. Máximo 8 MB.'); return }
    if (!['image/png', 'image/jpeg', 'image/bmp', 'image/tiff'].includes(f.type) && !/\.(png|jpe?g|bmp|tiff?)$/i.test(f.name)) {
      setError('Selecciona una imagen PNG, JPG, BMP o TIFF.'); return
    }
    if (vista) URL.revokeObjectURL(vista)
    setArchivo(f); setVista(URL.createObjectURL(f))
  }
  async function analizar() {
    if (!archivo) return
    setCargando(true); setError(''); setResultado(null)
    try {
      const form = new FormData(); form.append('file', archivo)
      const res = await fetch('/api/predict', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'La API rechazó la imagen.')
      setResultado(data)
    } catch (e) { setError(e.message || 'Error de comunicación con el servidor.') }
    finally { setCargando(false) }
  }
  function limpiar() {
    if (vista) URL.revokeObjectURL(vista)
    setVista(null); setArchivo(null); setResultado(null); setError('')
    if (entrada.current) entrada.current.value = ''
  }

  return <div className="layout">
    <aside className="sidebar">
      <div className="brand"><span className="symbol">◈</span><div><strong>VISION 360</strong><small>INDUSTRIAL AI · LAB</small></div></div>
      <div className="nav-label">PLATAFORMA</div>
      <div className="nav active"><span>▦</span> Análisis de soldadura</div>
      <div className="nav"><span>◇</span> RIAWELC · Demo</div>
      <div className="nav-label lower">INFORMACIÓN</div>
      <div className="side-note">Modelo Random Forest entrenado con radiografías públicas. No evalúa imágenes reales de la cámara industrial.</div>
      <div className="bottom"><span className={'status-dot '+(estado === 'activo' ? 'ready' : '')}></span> API: {estado}<small>Fase II · Gestión de Proyectos IA</small></div>
    </aside>
    <main className="main">
      <header><div><div className="crumb">PROYECTOS / VISION 360 / FASE II</div><h1>Inspección inteligente de soldadura</h1><p>Clasificación académica de defectos en imágenes radiográficas</p></div><div className="header-pill">RIAWELC <span>·</span> Random Forest</div></header>
      <div className="disclaimer"><span>ⓘ</span><div><strong>Entorno de demostración académica</strong><br/>Las predicciones son experimentales y no sustituyen inspecciones industriales, radiografías certificadas ni revisión de personal especializado.</div></div>
      <div className="grid">
        <section className="card"><div className="card-heading"><div><div className="eyebrow">01 · ENTRADA</div><h2>Imagen de soldadura</h2></div><span className="soft-chip">PNG · JPG · BMP · TIFF</span></div>
          <input type="file" ref={entrada} hidden accept="image/png,image/jpeg,image/bmp,image/tiff" onChange={e => cambiarArchivo(e.target.files?.[0])}/>
          <div className={'drop '+(vista ? 'has-image' : '')} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();cambiarArchivo(e.dataTransfer.files[0])}} onClick={() => entrada.current?.click()}>
            {vista ? <><img src={vista} alt="Imagen radiográfica seleccionada"/><div className="image-hint">Clic para cambiar la imagen</div></> : <><div className="drop-icon">⬆</div><strong>Arrastra tu radiografía aquí</strong><p>O haz clic para seleccionar un archivo</p><small>Hasta 8 MB · No se conserva en el servidor</small></>}
          </div>
          <div className="file-row">{archivo ? <><span className="file-name">◧ {archivo.name}</span><span>{(archivo.size/1024).toFixed(1)} KB</span></> : <span>No se ha seleccionado ninguna imagen</span>}</div>
          <div className="actions"><button className="button-primary" disabled={!archivo || cargando || estado !== 'activo'} onClick={analizar}>{cargando ? 'Procesando...' : 'Analizar imagen →'}</button><button className="button-secondary" onClick={limpiar} disabled={cargando}>Limpiar</button></div>
          {error && <p role="alert" className="error">{error}</p>}
        </section>
        <section className="card result-card"><div className="card-heading"><div><div className="eyebrow">02 · RESULTADO</div><h2>Clasificación del modelo</h2></div><span className="soft-chip">4 categorías</span></div>
          {!resultado ? <div className="empty"><div className="empty-graphic">◎</div><strong>Esperando análisis</strong><p>Selecciona una imagen para visualizar la clasificación y sus probabilidades estimadas.</p></div> : <div className="prediction"><div className={'analysis-status '+(resultado.ambigua?'inconclusive':'complete')}>{resultado.estado_analisis}</div><div className="prediction-label">CLASIFICACIÓN PREDICHA</div><div className={'prediction-name '+(resultado.clase_id===3?'good':'warning')}>{resultado.clase}</div><div className="prediction-sub">{resultado.resultado}</div><div className="confidence">Mayor probabilidad estimada <strong>{(resultado.confianza_estimada*100).toFixed(1)}%</strong></div>{resultado.ambigua&&<p className="ambiguity-note">Las dos categorías principales están separadas por solo {(resultado.margen_probabilidades*100).toFixed(1)} puntos porcentuales. Se conserva la predicción del modelo, pero requiere revisión.</p>}<div className="bars">{resultado.probabilidades.map(p=><div className="prob" key={p.id}><div className="prob-title"><span>{p.clase}</span><strong>{(p.probabilidad*100).toFixed(1)}%</strong></div><div className="track"><div className={'bar '+(resultado.clase_id===p.id?'highlight':'')} style={{width:`${p.probabilidad*100}%`}}/></div></div>)}</div><p className="result-note">Modelo: {resultado.modelo}. Las probabilidades no están calibradas y no garantizan exactitud. La clasificación no constituye un dictamen de calidad.</p></div>}
        </section>
      </div>
      <section className="cards-row"><div className="mini"><span>MODELO</span><strong>Random Forest</strong><small>Clasificación multiclase</small></div><div className="mini"><span>PREPROCESAMIENTO</span><strong>64 × 64 · HOG</strong><small>324 características</small></div><div className="mini"><span>CATEGORÍAS</span><strong>04</strong><small>{clases.join(' · ')}</small></div><div className="mini"><span>FUENTE</span><strong>RIAWELC</strong><small>Dataset público de radiografías</small></div></section>
      <footer><span>Vision 360 · Proyecto de maestría · Fase II</span><span><a href="/api/docs" target="_blank">Documentación API ↗</a></span></footer>
    </main>
  </div>
}
createRoot(document.getElementById('root')).render(<App />)
