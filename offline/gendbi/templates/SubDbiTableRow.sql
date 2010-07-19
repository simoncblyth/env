DROP TABLE IF EXISTS `{{ cls }}`;
CREATE TABLE `{{ cls }}` (
  `SEQNO` int(11) NOT NULL,
  `ROW_COUNTER` int(11) NOT NULL auto_increment,
  {% for r in t %}`{{ r.name }}` {{ r.dbtype }} default NULL COMMENT '{{ r.description }}',
  {% endfor %}PRIMARY KEY  (`SEQNO`,`ROW_COUNTER`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;



