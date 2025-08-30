from __future__ import annotations

from typing import Optional
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, Text, Boolean, DateTime, UniqueConstraint
from sqlalchemy.sql import func


class Base(DeclarativeBase):
	pass


class User(Base):
	__tablename__ = "users"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())
	stripe_customer_id: Mapped[Optional[str]] = mapped_column(String)
	subscription_tier: Mapped[Optional[str]] = mapped_column(String)
	subscription_status: Mapped[Optional[str]] = mapped_column(String)
	subscription_expires_at: Mapped[Optional[str]] = mapped_column(DateTime)


class PriceData(Base):
	__tablename__ = "price_data"
	__table_args__ = (
		UniqueConstraint('symbol', 'timestamp', 'source', name='uix_symbol_timestamp_source'),
	)

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	symbol: Mapped[str] = mapped_column(String, index=True)
	timestamp: Mapped[str] = mapped_column(DateTime, index=True)
	open: Mapped[Optional[float]] = mapped_column(Float)
	high: Mapped[Optional[float]] = mapped_column(Float)
	low: Mapped[Optional[float]] = mapped_column(Float)
	close: Mapped[float] = mapped_column(Float, nullable=False)
	volume: Mapped[Optional[int]] = mapped_column(Integer)
	source: Mapped[str] = mapped_column(String, nullable=False)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())


class NewsEvent(Base):
	__tablename__ = "news_events"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	symbol: Mapped[str] = mapped_column(String, index=True)
	headline: Mapped[str] = mapped_column(Text)
	url: Mapped[Optional[str]] = mapped_column(Text)
	source: Mapped[Optional[str]] = mapped_column(String)
	published_at: Mapped[Optional[str]] = mapped_column(DateTime, index=True)
	content_snippet: Mapped[Optional[str]] = mapped_column(Text)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())


class NewsAnalysis(Base):
	__tablename__ = "news_analysis"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	news_event_id: Mapped[int] = mapped_column(Integer)
	event_type: Mapped[Optional[str]] = mapped_column(String)
	severity: Mapped[Optional[str]] = mapped_column(String)
	time_horizon: Mapped[Optional[str]] = mapped_column(String)
	company_specific: Mapped[Optional[bool]] = mapped_column(Boolean)
	revenue_impact_pct: Mapped[Optional[float]] = mapped_column(Float)
	earnings_impact_pct: Mapped[Optional[float]] = mapped_column(Float)
	credibility_score: Mapped[Optional[float]] = mapped_column(Float)
	novelty_score: Mapped[Optional[float]] = mapped_column(Float)
	analysis_json: Mapped[Optional[str]] = mapped_column(Text)
	llm_model: Mapped[Optional[str]] = mapped_column(String)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())


class MarketIndicator(Base):
	__tablename__ = "market_indicators"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	indicator_name: Mapped[str] = mapped_column(String, index=True)
	timestamp: Mapped[str] = mapped_column(DateTime, index=True)
	value: Mapped[float] = mapped_column(Float)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())


class ComputedFeatures(Base):
	__tablename__ = "computed_features"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	symbol: Mapped[str] = mapped_column(String, index=True)
	timestamp: Mapped[str] = mapped_column(DateTime, index=True)
	feature_set: Mapped[str] = mapped_column(String, index=True)
	features_json: Mapped[str] = mapped_column(Text)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())


class TradingSignal(Base):
	__tablename__ = "trading_signals"

	id: Mapped[str] = mapped_column(String, primary_key=True)
	symbol: Mapped[str] = mapped_column(String, index=True)
	signal_type: Mapped[str] = mapped_column(String)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())
	expires_at: Mapped[Optional[str]] = mapped_column(DateTime)
	entry_price: Mapped[float] = mapped_column(Float)
	current_price: Mapped[float] = mapped_column(Float)
	target_price: Mapped[float] = mapped_column(Float)
	stop_loss_price: Mapped[float] = mapped_column(Float)
	position_size_pct: Mapped[Optional[float]] = mapped_column(Float)
	confidence_score: Mapped[float] = mapped_column(Float)
	oversold_score: Mapped[float] = mapped_column(Float)
	news_score: Mapped[Optional[float]] = mapped_column(Float)
	ml_prediction: Mapped[Optional[float]] = mapped_column(Float)
	triggering_event_id: Mapped[Optional[int]] = mapped_column(Integer)
	analysis_summary: Mapped[Optional[str]] = mapped_column(Text)
	features_json: Mapped[Optional[str]] = mapped_column(Text)
	status: Mapped[Optional[str]] = mapped_column(String, index=True)
	closed_at: Mapped[Optional[str]] = mapped_column(DateTime)
	exit_price: Mapped[Optional[float]] = mapped_column(Float)
	realized_pnl_pct: Mapped[Optional[float]] = mapped_column(Float)


class SignalPerformance(Base):
	__tablename__ = "signal_performance"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	signal_id: Mapped[str] = mapped_column(String, index=True)
	timestamp: Mapped[str] = mapped_column(DateTime, index=True)
	price: Mapped[float] = mapped_column(Float)
	unrealized_pnl_pct: Mapped[float] = mapped_column(Float)
	days_held: Mapped[int] = mapped_column(Integer)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())


class UserAlert(Base):
	__tablename__ = "user_alerts"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	user_id: Mapped[int] = mapped_column(Integer, index=True)
	alert_type: Mapped[Optional[str]] = mapped_column(String)
	destination: Mapped[str] = mapped_column(Text)
	filters_json: Mapped[Optional[str]] = mapped_column(Text)
	is_active: Mapped[Optional[bool]] = mapped_column(Boolean, default=True)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())


class ApiUsage(Base):
	__tablename__ = "api_usage"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	user_id: Mapped[int] = mapped_column(Integer, index=True)
	endpoint: Mapped[str] = mapped_column(String)
	timestamp: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp(), index=True)
	response_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
	status_code: Mapped[Optional[int]] = mapped_column(Integer)



class InterestingLoserDB(Base):
	__tablename__ = "interesting_losers"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	batch_id: Mapped[str] = mapped_column(String, index=True)
	symbol: Mapped[str] = mapped_column(String, index=True)
	price: Mapped[Optional[float]] = mapped_column(Float)
	change: Mapped[Optional[float]] = mapped_column(Float)
	change_percent: Mapped[Optional[float]] = mapped_column(Float)
	volume: Mapped[Optional[int]] = mapped_column(Integer)
	reason: Mapped[Optional[str]] = mapped_column(Text)
	rank: Mapped[Optional[int]] = mapped_column(Integer, index=True)
	session: Mapped[Optional[str]] = mapped_column(String)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp(), index=True)


class JobRecord(Base):
	__tablename__ = "jobs"

	id: Mapped[str] = mapped_column(String, primary_key=True)
	type: Mapped[str] = mapped_column(String)
	status: Mapped[str] = mapped_column(String, index=True)
	created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp(), index=True)
	updated_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.current_timestamp())
	result_json: Mapped[Optional[str]] = mapped_column(Text)
	error: Mapped[Optional[str]] = mapped_column(Text)
	progress: Mapped[Optional[float]] = mapped_column(Float)
	expires_at: Mapped[Optional[str]] = mapped_column(DateTime, index=True)

