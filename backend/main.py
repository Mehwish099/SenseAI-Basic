"""SenseAI local API. Gloss ranking uses contextual transformer embeddings when available."""
from __future__ import annotations
import csv, hashlib, hmac, io, logging, os, re, secrets
from datetime import datetime, timezone
from typing import Optional
import nltk
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("senseai")
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./senseai.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
Session = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(254), nullable=False, unique=True, index=True)
    password_salt = Column(String(64), nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

class LoginSession(Base):
    __tablename__ = "login_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(Integer, nullable=False)

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    sentence = Column(Text, nullable=False)
    target = Column(String(80), nullable=False)
    result = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    user_id = Column(Integer, nullable=True, index=True)
Base.metadata.create_all(engine)
with engine.connect() as _connection:
    if DB_URL.startswith("sqlite"):
        columns = {row[1] for row in _connection.exec_driver_sql("PRAGMA table_info(analyses)").fetchall()}
        if "user_id" not in columns:
            _connection.exec_driver_sql("ALTER TABLE analyses ADD COLUMN user_id INTEGER")
            _connection.commit()

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    target: str = Field(min_length=1, max_length=80)
    occurrence: int = Field(default=1, ge=1)
class BatchRequest(BaseModel):
    items: list[AnalyzeRequest] = Field(min_length=1, max_length=100)
class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)
class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)

app = FastAPI(title="SenseAI API", version="1.0.0", description="Contextual word sense analysis backed by WordNet glosses.")
# Keep local development ports working, and permit the exact deployed frontend origin in production.
_frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")
app.add_middleware(CORSMiddleware, allow_origins=[_frontend_origin] if _frontend_origin else [], allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$", allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "Authorization"])
_bearer = HTTPBearer(auto_error=False)

def _password_hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000).hex()

def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Sign in to continue.", headers={"WWW-Authenticate": "Bearer"})
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    db = Session()
    try:
        session = db.query(LoginSession).filter(LoginSession.token_hash == token_hash).first()
        if session is None or session.expires_at <= int(datetime.now(timezone.utc).timestamp()):
            raise HTTPException(401, "Your session has expired. Please sign in again.", headers={"WWW-Authenticate": "Bearer"})
        user = db.query(User).filter(User.id == session.user_id).first()
        if user is None: raise HTTPException(401, "Account not found. Please sign in again.")
        return user
    finally: db.close()

def _new_session(db, user: User) -> dict:
    token = secrets.token_urlsafe(32)
    expires = int(datetime.now(timezone.utc).timestamp()) + 12 * 60 * 60
    db.add(LoginSession(user_id=user.id, token_hash=hashlib.sha256(token.encode()).hexdigest(), expires_at=expires))
    db.commit()
    return {"access_token": token, "token_type": "bearer", "email": user.email, "expires_at": expires}
_encoder = None
_model_status = "not_loaded"
_model_attempted = False

def _wordnet():
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("bank")
        return wn
    except LookupError:
        for path in ("C:/nltk_data", os.path.expanduser("~/nltk_data")):
            nltk.data.path.append(path)
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("bank")
            return wn
        except LookupError:
            return None

def _encoder_model():
    global _encoder, _model_status, _model_attempted
    if _encoder is not None: return _encoder
    if _model_attempted: return None
    _model_attempted = True
    if os.getenv("SENSEAI_DISABLE_TRANSFORMER", "").lower() in ("1", "true", "yes"):
        _model_status = "lexical_fallback"
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(os.getenv("SENSEAI_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        _model_status = "transformer"
        return _encoder
    except Exception as exc:
        log.warning("Transformer unavailable, using clearly labeled gloss overlap: %s", exc)
        _model_status = "lexical_fallback"
        return None

def _overlap(context: str, gloss: str) -> float:
    stop = {"the","a","an","is","was","are","were","to","of","in","on","and","or","it","with","for","as","at","by","this","that","from","be","after","before","into","his","her","their"}
    words = lambda s: {w.lower() for w in re.findall(r"[A-Za-z]+", s) if w.lower() not in stop}
    c, g = words(context), words(gloss)
    return len(c & g) / max(1, len(c | g))

def analyze(payload: AnalyzeRequest, user_id: int | None = None) -> dict:
    text, target = payload.text.strip(), payload.target.strip()
    if not text: raise HTTPException(422, "Enter a sentence or paragraph to analyze.")
    pattern = re.compile(r"(?<!\w)" + re.escape(target) + r"(?!\w)", re.I)
    matches = list(pattern.finditer(text))
    if not matches: raise HTTPException(422, f"The target ‘{target}’ was not found in your text.")
    if payload.occurrence > len(matches): raise HTTPException(422, f"Occurrence {payload.occurrence} does not exist; this target appears {len(matches)} time(s).")
    match = matches[payload.occurrence - 1]
    surface = match.group(0)
    wn = _wordnet()
    if wn is None:
        raise HTTPException(503, "WordNet data is not installed. Run: python -m nltk.downloader wordnet omw-1.4")
    lemma = surface.lower()
    synsets = wn.synsets(lemma)
    if not synsets:
        # Common inflections through WordNet's morphology.
        for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
            root = wn.morphy(lemma, pos)
            if root:
                synsets = wn.synsets(root, pos=pos)
                if synsets: break
    if not synsets:
        return {"status":"unknown", "target":surface, "message":f"No WordNet senses were found for ‘{surface}’. Try the base form or another target.", "candidates":[], "method":"WordNet lookup"}
    encoder = _encoder_model()
    context = text[max(0, match.start()-240):min(len(text), match.end()+240)]
    glosses = [s.definition() + ". Examples: " + "; ".join(s.examples()[:2]) for s in synsets]
    if encoder:
        vectors = encoder.encode([context] + glosses, normalize_embeddings=True)
        scores = (vectors[1:] @ vectors[0]).tolist()
        method = "Contextual transformer gloss similarity"
        score_label = "Cosine similarity between the surrounding context and each WordNet definition; it is a ranking signal, not calibrated probability."
    else:
        scores = [_overlap(context, g) for g in glosses]
        method = "WordNet gloss word overlap (fallback; not a transformer prediction)"
        score_label = "Jaccard overlap between context words and definition/example words; it is a weak heuristic, not a probability."
    ranked = sorted(zip(synsets, scores), key=lambda x: x[1], reverse=True)[:12]
    top_s, top_score = ranked[0]
    sorted_scores = sorted(scores, reverse=True)
    gap = top_score - (sorted_scores[1] if len(sorted_scores)>1 else 0)
    confidence = "uncertain" if top_score < (0.22 if encoder else 0.035) or (len(synsets)>1 and gap < (0.015 if encoder else 0.01)) else "tentative"
    if top_score >= (0.38 if encoder else 0.12) and gap >= (0.04 if encoder else 0.04): confidence = "stronger match"
    def sense(s, score):
        return {"synset":s.name(),"pos":s.pos(),"definition":s.definition(),"examples":s.examples()[:2],"lemmas":[l.name().replace("_"," ") for l in s.lemmas()[:8]],"score":round(float(score),4),"hypernyms":[h.name() for h in s.hypernyms()[:4]],"selected":s==top_s}
    result = {"status":"ok", "target":surface, "sentence":text, "start":match.start(), "end":match.end(), "occurrence":payload.occurrence, "occurrences":len(matches), "prediction":sense(top_s,top_score), "candidates":[sense(s,score) for s,score in ranked], "confidence":confidence,"score_explanation":score_label,"method":method,"context":context}
    db=Session()
    try:
        import json
        db.add(Analysis(sentence=text,target=surface,result=json.dumps(result),created_at=datetime.now(timezone.utc),user_id=user_id))
        db.commit()
    finally: db.close()
    return result

@app.post("/api/auth/register")
def register(payload: RegisterRequest):
    email = payload.email.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email): raise HTTPException(422, "Enter a valid email address.")
    db = Session()
    try:
        if db.query(User).filter(User.email == email).first(): raise HTTPException(409, "An account with this email already exists. Sign in instead.")
        salt = secrets.token_bytes(24)
        user = User(email=email, password_salt=salt.hex(), password_hash=_password_hash(payload.password, salt))
        db.add(user); db.commit(); db.refresh(user)
        return _new_session(db, user)
    finally: db.close()

@app.post("/api/auth/login")
def login(payload: LoginRequest):
    email = payload.email.strip().lower()
    db = Session()
    try:
        user = db.query(User).filter(User.email == email).first()
        salt = bytes.fromhex(user.password_salt) if user else b"senseai-invalid-user-salt"
        valid = hmac.compare_digest(_password_hash(payload.password, salt), user.password_hash if user else "0" * 64)
        if not valid: raise HTTPException(401, "Email or password doesn’t match. Try again.")
        return _new_session(db, user)
    finally: db.close()

@app.get("/api/auth/me")
def auth_me(user: User = Depends(get_current_user)):
    return {"email": user.email}

@app.post("/api/auth/logout")
def logout(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer), user: User = Depends(get_current_user)):
    if credentials:
        token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        db = Session()
        try: db.query(LoginSession).filter(LoginSession.token_hash == token_hash, LoginSession.user_id == user.id).delete(); db.commit()
        finally: db.close()
    return {"status": "signed_out"}

@app.get("/api/health")
def health():
    wn=_wordnet()
    return {"status":"ok","wordnet":"ready" if wn else "missing","model":_model_status,"model_name":os.getenv("SENSEAI_MODEL","sentence-transformers/all-MiniLM-L6-v2")}
@app.post("/api/wsd/analyze")
def analyze_endpoint(payload: AnalyzeRequest, user: User = Depends(get_current_user)): return analyze(payload, user.id)
@app.post("/api/wsd/batch")
def batch_endpoint(payload: BatchRequest, user: User = Depends(get_current_user)):
    rows=[]
    for item in payload.items:
        try: rows.append(analyze(item, user.id))
        except HTTPException as exc: rows.append({"status":"error","target":item.target,"sentence":item.text,"message":exc.detail})
    return {"results":rows,"count":len(rows)}
@app.get("/api/wsd/history")
def history_endpoint(limit: int=50, user: User = Depends(get_current_user)):
    import json
    db=Session()
    try:
        rows=db.query(Analysis).filter(Analysis.user_id==user.id).order_by(Analysis.created_at.desc()).limit(max(1,min(200,limit))).all()
        return [{"id":r.id,"sentence":r.sentence,"target":r.target,"created_at":r.created_at.isoformat(),"result":json.loads(r.result)} for r in rows]
    finally: db.close()
@app.delete("/api/wsd/history")
def clear_history(user: User = Depends(get_current_user)):
    db=Session()
    try: count=db.query(Analysis).filter(Analysis.user_id==user.id).delete(); db.commit(); return {"deleted":count}
    finally: db.close()

class EvalRequest(BaseModel):
    items: list[AnalyzeRequest] = Field(min_length=1, max_length=500)
    labels: list[str] = Field(min_length=1, max_length=500)
    dataset: str = Field(default="User-provided labels", max_length=120)
@app.post("/api/evaluation/run")
def evaluation(payload: EvalRequest, user: User = Depends(get_current_user)):
    if len(payload.items)!=len(payload.labels): raise HTTPException(422,"Provide one gold synset label per sentence.")
    predicted=[]; errors=[]
    for item in payload.items:
        try: predicted.append(analyze(item, user.id).get("prediction",{}).get("synset"))
        except HTTPException as exc: predicted.append(None); errors.append(exc.detail)
    gold=payload.labels
    evaluated=sum(p is not None for p in predicted)
    classes=sorted(set(gold)|{p for p in predicted if p is not None})
    matrix={actual:{guess:0 for guess in classes} for actual in classes}
    for actual,guess in zip(gold,predicted):
        if guess is not None: matrix[actual][guess]+=1
    per_class={}
    for cls in classes:
        tp=matrix[cls][cls]
        fp=sum(matrix[a][cls] for a in classes if a!=cls)
        fn=sum(matrix[cls][g] for g in classes if g!=cls)
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        per_class[cls]={"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,"support":sum(matrix[cls].values())}
    macro_f1=sum(m["f1"] for m in per_class.values())/len(per_class) if per_class else None
    correct=sum(p==g for p,g in zip(predicted,gold) if p is not None)
    return {"dataset":payload.dataset,"sample_count":len(payload.items),"evaluated_count":evaluated,"accuracy":correct/evaluated if evaluated else None,"macro_f1":macro_f1,"per_class":per_class,"confusion_matrix":{"labels":classes,"values":matrix},"predictions":predicted,"labels":gold,"errors":errors,"limitations":"Metrics use only the supplied labeled examples. Results are not representative unless the labels come from a documented held-out dataset. Precision, recall, and F1 are computed per synset; macro-F1 averages all listed classes."}
